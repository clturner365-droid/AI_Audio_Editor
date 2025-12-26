# resume.py
# Handles reading and writing the progress checkpoint (Progress.txt)

import os

PROGRESS_FILE = "Progress.txt"

def get_last_completed_index():
    """
    Reads Progress.txt and returns the last successfully processed index.
    If the file doesn't exist or is invalid, return -1 (meaning start from the top).
    """
    if not os.path.exists(PROGRESS_FILE):
        return -1

    try:
        with open(PROGRESS_FILE, "r") as f:
            value = f.read().strip()
            return int(value)
    except:
        # Corrupted or unreadable progress file → start from scratch
        return -1


def update_progress(index):
    """
    Writes the given index to Progress.txt.
    This is called ONLY after a file is fully processed.
    """
    try:
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(index))
    except Exception as e:
        # If this fails, the system should log it but continue running.
        # Logging will be handled by logging_system.py
        pass
