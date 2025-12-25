# logging_system.py
# Centralized logging for Pipeline B

import os
from datetime import datetime

LOG_DIR = os.path.join("PipelineB", "Logs")

MASTER_REVIEW_LOG = os.path.join(LOG_DIR, "MasterReview.log")
MISSING_FILES_LOG = os.path.join(LOG_DIR, "MissingFiles.log")
FINGERPRINT_ERROR_LOG = os.path.join(LOG_DIR, "FingerprintErrors.log")
SYSTEM_LOG = os.path.join(LOG_DIR, "System.log")


def ensure_log_dir():
    """
    Ensures the Logs/ directory exists.
    Called once at startup by main.py.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)


def timestamp():
    """
    Returns a clean timestamp for log entries.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------
# SYSTEM LOGGING
# ---------------------------------------------------------

def log_system(message):
    """
    Writes a message to the system log.
    """
    line = f"[{timestamp()}] {message}\n"
    with open(SYSTEM_LOG, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------
# MASTER REVIEW LOG
# ---------------------------------------------------------

def log_review(message):
    """
    Writes a message to the master review log.
    Used for anything requiring human attention.
    """
    line = f"[{timestamp()}] {message}\n"
    with open(MASTER_REVIEW_LOG, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------
# MISSING FILES LOG
# ---------------------------------------------------------

def log_missing_file(path):
    """
    Logs missing input files.
    """
    line = f"[{timestamp()}] MISSING FILE: {path}\n"
    with open(MISSING_FILES_LOG, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------
# FINGERPRINT ERROR LOG
# ---------------------------------------------------------

def log_fingerprint_error(message):
    """
    Logs fingerprint-related issues.
    """
    line = f"[{timestamp()}] {message}\n"
    with open(FINGERPRINT_ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------
# PER-FILE LOGGING
# ---------------------------------------------------------

def start_file_log(path):
    """
    Creates a new per-file log buffer.
    Returns a list that other modules append to.
    """
    return [f"Processing file: {path}", f"Start time: {timestamp()}", ""]


def append_file_log(log_buffer, message):
    """
    Adds a line to the per-file log buffer.
    """
    log_buffer.append(f"[{timestamp()}] {message}")


def write_file_log(log_path, log_buffer):
    """
    Writes the per-file log buffer to disk.
    """
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_buffer))
