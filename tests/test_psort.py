"""Tests for psort task slicing and register_in_db wiring.

Plaso is a heavy import-time dep that isn't necessarily installed in the
local pytest environment. We stub the plaso modules psort.py imports at
collection time so the tests run anywhere.
"""

import base64
import pytest
import json
import sys
import types

from src import psort  # noqa: E402
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

if "plaso" not in sys.modules:
    plaso_pkg = types.ModuleType("plaso")
    plaso_output_pkg = types.ModuleType("plaso.output")
    plaso_output_manager = types.ModuleType("plaso.output.manager")

    class _FakeOutputManager:
        @staticmethod
        def GetOutputClasses():
            return [("csv", None), ("jsonl", None), ("l2tcsv", None)]

    plaso_output_manager.OutputManager = _FakeOutputManager
    plaso_output_pkg.manager = plaso_output_manager
    plaso_pkg.output = plaso_output_pkg
    sys.modules["plaso"] = plaso_pkg
    sys.modules["plaso.output"] = plaso_output_pkg
    sys.modules["plaso.output.manager"] = plaso_output_manager

# Stub openrelik_common — psort.py imports telemetry + Logger from it.
if "openrelik_common" not in sys.modules:
    openrelik_common_pkg = types.ModuleType("openrelik_common")
    telemetry_mod = types.ModuleType("openrelik_common.telemetry")
    telemetry_mod.add_attribute_to_current_span = lambda *a, **k: None
    logging_mod = types.ModuleType("openrelik_common.logging")

    class _FakeLogger:
        def get_logger(self, *a, **k):
            return MagicMock()

        def bind(self, *a, **k):
            pass

    logging_mod.Logger = _FakeLogger
    openrelik_common_pkg.telemetry = telemetry_mod
    openrelik_common_pkg.logging = logging_mod
    sys.modules["openrelik_common"] = openrelik_common_pkg
    sys.modules["openrelik_common.telemetry"] = telemetry_mod
    sys.modules["openrelik_common.logging"] = logging_mod

# Stub the celery app import (src.app -> app.py instantiates Celery() at
# import time and pulls in redis/etc.). We replace it with a minimal shim
# whose `celery.task` decorator is a no-op pass-through.
if "src.app" not in sys.modules:
    app_mod = types.ModuleType("src.app")

    class _FakeCelery:
        def task(self, *dargs, **dkwargs):
            def deco(fn):
                return fn

            return deco

    app_mod.celery = _FakeCelery()
    sys.modules["src.app"] = app_mod


class TestComputeSliceRanges:
    def test_single_slice_returns_sentinel(self):
        ranges = psort._compute_slice_ranges(
            1, 3, datetime(2026, 5, 20, tzinfo=timezone.utc)
        )
        assert ranges == [(None, None)]

    def test_three_slices_three_months_each(self):
        now = datetime(2026, 5, 20, tzinfo=timezone.utc)
        ranges = psort._compute_slice_ranges(3, 3, now)
        assert len(ranges) == 3
        # Oldest first.
        for start, end in ranges:
            assert start < end
        for prev, curr in zip(ranges, ranges[1:]):
            # Each slice's end equals the next slice's start (no overlap, no gap).
            assert prev[1] == curr[0]
        # Newest slice ends at now.
        assert ranges[-1][1] == now
        # Spans 9 months total.
        assert (ranges[-1][1] - ranges[0][0]).days >= 9 * 28

    def test_four_slices_four_months_each(self):
        now = datetime(2026, 5, 20, tzinfo=timezone.utc)
        ranges = psort._compute_slice_ranges(4, 4, now)
        assert len(ranges) == 4
        assert ranges[-1][1] == now


class TestParseIntInRange:
    def test_default_when_none(self):
        assert psort._parse_int_in_range(None, "x", 1, 12, default=3) == 3

    def test_default_when_empty_string(self):
        assert psort._parse_int_in_range("", "x", 1, 12, default=3) == 3

    def test_valid_int(self):
        assert psort._parse_int_in_range("4", "x", 1, 12, default=1) == 4
        assert psort._parse_int_in_range(7, "x", 1, 12, default=1) == 7

    def test_below_range(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            psort._parse_int_in_range("0", "slices", 1, 12, default=1)

    def test_above_range(self):
        with pytest.raises(ValueError, match="between 1 and 12"):
            psort._parse_int_in_range("13", "slices", 1, 12, default=1)

    def test_non_integer(self):
        with pytest.raises(ValueError, match="must be an integer"):
            psort._parse_int_in_range("foo", "slices", 1, 12, default=1)


def _decode(result_b64: str) -> dict:
    return json.loads(base64.b64decode(result_b64.encode("utf-8")).decode("utf-8"))


@pytest.fixture
def fake_subprocess(tmp_path, monkeypatch):
    """Patch subprocess.Popen + os.path.exists so the task body runs without
    actually invoking psort. Returns a list capturing every command invoked.
    """
    invoked: list[list[str]] = []

    class _FakeProcess:
        def __init__(self, *args, **kwargs):
            self._polled = False
            self.stdout = MagicMock()
            self.stdout.read.return_value = ""
            self.stderr = MagicMock()
            self.stderr.read.return_value = ""

        def poll(self):
            # Return None once (so the loop body can run if it wants), then 0.
            if not self._polled:
                self._polled = True
                return 0
            return 0

    def fake_popen(cmd, *args, **kwargs):
        invoked.append(list(cmd))
        return _FakeProcess()

    monkeypatch.setattr(psort.subprocess, "Popen", fake_popen)
    # status_file never exists, so the inner status-polling loop body is
    # skipped (process.poll() returns 0 immediately anyway).
    monkeypatch.setattr(psort.os.path, "exists", lambda p: False)
    return invoked


@pytest.fixture
def bound_task():
    """Build a stand-in for the bound `self` Celery would pass."""
    task = MagicMock()
    task.send_event = MagicMock()
    return task


def test_psort_no_slices_backwards_compatible(tmp_path, fake_subprocess, bound_task):
    """slices unset → one psort invocation, no --filter, today's filename."""
    inputs = [{"path": "/in/foo.plaso", "display_name": "foo.plaso"}]

    raw = psort.psort(
        bound_task,
        pipe_result=None,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-1",
        task_config=None,
    )

    assert len(fake_subprocess) == 1
    cmd = fake_subprocess[0]
    assert "--filter" not in cmd
    result = _decode(raw)
    assert len(result["output_files"]) == 1
    assert result["output_files"][0]["display_name"] == "foo.plaso.csv"
    assert result["output_files"][0]["register_in_db"] is True


def test_psort_three_slices_emits_three_filtered_runs(
    tmp_path, fake_subprocess, bound_task
):
    inputs = [{"path": "/in/foo.plaso", "display_name": "foo.plaso"}]

    raw = psort.psort(
        bound_task,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-2",
        task_config={"slices": "3", "months_per_slice": "3"},
    )

    assert len(fake_subprocess) == 3
    for cmd in fake_subprocess:
        assert "--filter" in cmd
        filter_arg = cmd[cmd.index("--filter") + 1]
        assert "date >" in filter_arg and "date <=" in filter_arg
    result = _decode(raw)
    display_names = sorted(f["display_name"] for f in result["output_files"])
    assert display_names == [
        "foo.plaso.slice-1-of-3.csv",
        "foo.plaso.slice-2-of-3.csv",
        "foo.plaso.slice-3-of-3.csv",
    ]


def test_psort_register_in_db_false_propagates(tmp_path, fake_subprocess, bound_task):
    inputs = [{"path": "/in/foo.plaso", "display_name": "foo.plaso"}]

    raw = psort.psort(
        bound_task,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-3",
        task_config={"register_in_db": False},
    )
    result = _decode(raw)
    assert result["output_files"][0]["register_in_db"] is False


def test_psort_invalid_slices_raises_before_subprocess(
    tmp_path, fake_subprocess, bound_task
):
    inputs = [{"path": "/in/foo.plaso", "display_name": "foo.plaso"}]

    with pytest.raises(ValueError, match="slices"):
        psort.psort(
            bound_task,
            input_files=inputs,
            output_path=str(tmp_path),
            workflow_id="wf-x",
            task_config={"slices": "13"},
        )
    assert fake_subprocess == []


def test_psort_invalid_slices_non_integer_raises(tmp_path, fake_subprocess, bound_task):
    inputs = [{"path": "/in/foo.plaso", "display_name": "foo.plaso"}]

    with pytest.raises(ValueError, match="must be an integer"):
        psort.psort(
            bound_task,
            input_files=inputs,
            output_path=str(tmp_path),
            workflow_id="wf-x",
            task_config={"slices": "foo"},
        )
    assert fake_subprocess == []


def test_psort_multiple_inputs_emit_all_outputs(tmp_path, fake_subprocess, bound_task):
    """Regression for the prior bug where output_files.append sat outside the
    per-input loop and only the last input's file was reported."""
    inputs = [
        {"path": "/in/a.plaso", "display_name": "a.plaso"},
        {"path": "/in/b.plaso", "display_name": "b.plaso"},
    ]

    raw = psort.psort(
        bound_task,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-multi",
        task_config=None,
    )
    result = _decode(raw)
    names = sorted(f["display_name"] for f in result["output_files"])
    assert names == ["a.plaso.csv", "b.plaso.csv"]


def test_psort_threads_original_path_from_input(tmp_path, fake_subprocess, bound_task):
    """Each output file's original_path must reference the upstream upload so
    downstream tasks (e.g. S3 export) can name objects after the user's file."""
    inputs = [
        {
            "path": "/in/foo.plaso",
            "display_name": "foo.plaso",
            "original_path": "/uploads/evidence.E01",
        }
    ]

    raw = psort.psort(
        bound_task,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-original",
        task_config=None,
    )
    result = _decode(raw)
    assert result["output_files"][0]["original_path"] == "/uploads/evidence.E01"


def test_psort_falls_back_to_path_when_original_path_missing(
    tmp_path, fake_subprocess, bound_task
):
    inputs = [{"path": "/in/foo.plaso", "display_name": "foo.plaso"}]

    raw = psort.psort(
        bound_task,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-fallback",
        task_config=None,
    )
    result = _decode(raw)
    assert result["output_files"][0]["original_path"] == "/in/foo.plaso"


def test_psort_slices_x_inputs_cartesian(tmp_path, fake_subprocess, bound_task):
    inputs = [
        {"path": "/in/a.plaso", "display_name": "a.plaso"},
        {"path": "/in/b.plaso", "display_name": "b.plaso"},
    ]

    raw = psort.psort(
        bound_task,
        input_files=inputs,
        output_path=str(tmp_path),
        workflow_id="wf-cartesian",
        task_config={"slices": "2", "months_per_slice": "3"},
    )
    assert len(fake_subprocess) == 4  # 2 inputs × 2 slices
    result = _decode(raw)
    assert len(result["output_files"]) == 4
