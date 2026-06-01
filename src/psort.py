# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import subprocess
import time
from datetime import datetime, timezone

from celery import signals
from celery.utils.log import get_task_logger
from dateutil.relativedelta import relativedelta
from openrelik_common import telemetry
from openrelik_common.logging import Logger
from openrelik_worker_common.file_utils import create_output_file
from openrelik_worker_common.task_utils import create_task_result, get_input_files
from plaso.output import manager as output_manager

from .app import celery
from .utils import log2timeline_status_to_dict, process_plaso_cli_logs

# Get the supported psort output formats.
output_formats_available = {
    name for name, _ in output_manager.OutputManager.GetOutputClasses()
}

# Task name used to register and route the task to the correct queue.
TASK_NAME = "openrelik-worker-plaso.tasks.psort"

# Task metadata for registration in the core system.
TASK_METADATA = {
    "display_name": "Plaso Psort",
    "description": "Process Plaso storage files",
    "task_config": [
        {
            "name": "output_format",
            "label": "Select output format to use",
            "description": "Select the output format for psort, default will be csv.",
            "type": "select",
            "items": sorted(output_formats_available),
            "required": False,
        },
        {
            "name": "slices",
            "label": "Number of time slices (1 = no slicing)",
            "description": (
                "Default 1 produces a single output file covering the entire "
                "storage (no date filter). Set to N > 1 to "
                "split the output into N files, each covering a trailing "
                "window of 'Months per slice' months ending at the previous "
                "slice's start. Allowed range: 1-12."
            ),
            "type": "text",
            "value": "1",
            "required": False,
        },
        {
            "name": "months_per_slice",
            "label": "Months per slice (ignored when slices = 1)",
            "description": (
                "Width of each slice in months. Only takes effect when "
                "slices > 1; with slices = 1 the entire storage is exported "
                "and this value is unused. Slice i covers "
                "[now - i*M months, now - (i-1)*M months). Allowed range: 1-12."
            ),
            "type": "text",
            "value": "3",
            "required": False,
        },
        {
            "name": "register_in_db",
            "label": "Register output files in the database",
            "description": (
                "When enabled (default), each psort output file is registered "
                "in the OpenRelik database and appears in the UI. Disable for "
                "intermediate runs that only feed downstream tasks."
            ),
            "type": "switch",
            "value": True,
            "required": False,
        },
    ],
}


def _compute_slice_ranges(
    slices: int, months_per_slice: int, now: datetime
) -> list[tuple[datetime | None, datetime | None]]:
    """Return slice (start, end) pairs, oldest-first.

    For ``slices == 1`` returns ``[(None, None)]`` as a sentinel meaning
    "no filter — run psort once over the whole storage file".
    """
    if slices == 1:
        return [(None, None)]
    ranges: list[tuple[datetime, datetime]] = []
    for i in range(slices, 0, -1):
        end = now - relativedelta(months=(i - 1) * months_per_slice)
        start = now - relativedelta(months=i * months_per_slice)
        ranges.append((start, end))
    return ranges


def _parse_int_in_range(raw, name: str, lo: int, hi: int, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"task_config[{name!r}] must be an integer; got {raw!r}") from exc
    if value < lo or value > hi:
        raise ValueError(
            f"task_config[{name!r}] must be between {lo} and {hi}; got {value}"
        )
    return value

log_root = Logger()
logger = log_root.get_logger(__name__, get_task_logger(__name__))


@signals.task_prerun.connect
def on_task_prerun(sender, task_id, task, args, kwargs, **_):
    log_root.bind(
        task_id=task_id,
        task_name=task.name,
        worker_name=TASK_METADATA.get("display_name"),
    )


@celery.task(bind=True, name=TASK_NAME, metadata=TASK_METADATA)
def psort(
    self,
    pipe_result: str = None,
    input_files: list = None,
    output_path: str = None,
    workflow_id: str = None,
    task_config: dict = None,
) -> str:
    """Run psort on input files.

    Args:
        pipe_result: Base64-encoded result from the previous Celery task, if any.
        input_files: List of input file dictionaries (unused if pipe_result exists).
        output_path: Path to the output directory.
        workflow_id: ID of the workflow.
        task_config: User configuration for the task.

    Returns:
        Base64-encoded dictionary containing task results.
    """
    log_root.bind(workflow_id=workflow_id)
    logger.info(f"Starting {TASK_NAME} for workflow {workflow_id}")

    input_files = get_input_files(pipe_result, input_files or [])
    output_files = []
    command_string = ""

    telemetry.add_attribute_to_current_span("input_files", input_files)
    telemetry.add_attribute_to_current_span("task_config", task_config)
    telemetry.add_attribute_to_current_span("workflow_id", workflow_id)

    # Set output extensions based on chosen task config output format, default is csv
    output_extension = "csv"
    if task_config and task_config.get("output_format"):
        output_extension = task_config["output_format"]

    cfg = task_config or {}
    register_in_db = cfg.get("register_in_db", True)
    slices = _parse_int_in_range(cfg.get("slices"), "slices", 1, 12, default=1)
    months_per_slice = _parse_int_in_range(
        cfg.get("months_per_slice"), "months_per_slice", 1, 12, default=3
    )
    slice_ranges = _compute_slice_ranges(
        slices, months_per_slice, datetime.now(timezone.utc)
    )

    psort_filter_fmt = "%Y-%m-%dT%H:%M:%S"

    total_slices = len(slice_ranges)
    for input_file in input_files:
        for slice_idx, (start, end) in enumerate(slice_ranges, start=1):
            if start is None and end is None:
                slice_display_name = (
                    f"{input_file.get('display_name')}.{output_extension}"
                )
            else:
                slice_display_name = (
                    f"{input_file.get('display_name')}."
                    f"slice-{slice_idx}-of-{total_slices}.{output_extension}"
                )

            output_file = create_output_file(
                output_path,
                display_name=slice_display_name,
                data_type=f"plaso:psort:{output_extension}",
                original_path=(
                    input_file.get("original_path") or input_file.get("path")
                ),
                register_in_db=register_in_db,
            )
            status_file = create_output_file(output_path, extension="status")

            command = [
                "psort.py",
                "--quiet",
                "--status-view",
                "file",
                "--additional_fields",
                "yara_match",
                "--status-view-file",
                status_file.path,
                "-w",
                output_file.path,
            ]
            if cfg.get("output_format"):
                command.extend(["-o", cfg["output_format"]])
            if start is not None and end is not None:
                command.extend(
                    [
                        "--filter",
                        (
                            f"date > '{start.strftime(psort_filter_fmt)}' "
                            f"AND date <= '{end.strftime(psort_filter_fmt)}'"
                        ),
                    ]
                )
            command.append(input_file.get("path"))

            command_string = " ".join(command)

            # Send initial status event to indicate slice start
            self.send_event(
                "task-progress",
                data={"slice": f"{slice_idx}/{slices}"} if slices > 1 else {},
            )

            logger.info(f"Starting {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            while process.poll() is None:
                if not os.path.exists(status_file.path):
                    continue
                with open(status_file.path, "r") as f:
                    status_dict = {}
                    try:
                        status_dict = log2timeline_status_to_dict(f.read())
                    except:
                        pass
                    if slices > 1:
                        status_dict["slice"] = f"{slice_idx}/{slices}"
                    self.send_event("task-progress", data=status_dict)
                time.sleep(2)
            logger.info(process.stdout.read())
            if process.stderr:
                process_plaso_cli_logs(process.stderr.read(), logger)

            output_files.append(output_file.to_dict())

    return create_task_result(
        output_files=output_files,
        workflow_id=workflow_id,
        command=command_string,
    )
