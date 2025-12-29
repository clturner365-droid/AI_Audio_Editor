#!/usr/bin/env python3
"""
modules/resume.py

Crash‑safe progress tracking for batch processing.
Provides:
    - get_last_completed_index()
    - update_progress(index)
    - run(state, ctx)  # optional dispatcher wrapper
"""

import os
import tempfile
from modules.logging_system import log_system, append_file_log

PROGRESS_FILE = "Progress.txt"


# ---------------------------------------------------------
# READ PROGRESS (with validation)
# ---------------------------------------------------------

def get_last_completed_index():
    """
    Reads Progress.txt and returns the last successfully processed index.
    If missing, unreadable, or invalid → return -1.
    """
    if not os.path.exists(PROGRESS_FILE):
        return -1

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            value = int(raw)

            # Validate: must be >= -1
            if value < -1:
                log_system(f"resume: invalid negative index '{value}', resetting to -1")
                return -1

            return value

    except Exception as e:
        log_system(f"resume: failed to read progress file: {e}")
        return -1


# ---------------------------------------------------------
# ATOMIC WRITE (with logging)
# ---------------------------------------------------------

def update_progress(index):
    """
    Writes the given index to Progress.txt atomically.
    Called ONLY after a file is fully processed.
    """
    try:
        dirpath = os.path.dirname(PROGRESS_FILE) or "."
        tmp = tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False, encoding="utf-8")
        tmp.write(str(index))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
        tmp.close()

        os.replace(tmp_path, PROGRESS_FILE)

    except Exception as e:
        # Log but do not crash the pipeline
        log_system(f"resume: failed to update progress file: {e}")


# ---------------------------------------------------------
# OPTIONAL DISPATCHER WRAPPER
# ---------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher-compatible wrapper.
    Called at the END of processing a file.
    Updates Progress.txt with ctx["file_index"].
    """
    log_buffer = ctx.get("log_buffer")
    file_index = ctx.get("file_index")

    append_file_log(log_buffer, f"resume: updating progress to index {file_index}")
    update_progress(file_index)

    return state
