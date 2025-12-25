# file_manager.py
# Handles file existence checks, output path building, and preventing overwrites.

import os

# Base output directory for processed files
PROCESSED_DIR = os.path.join("PipelineB", "Processed")

def ensure_processed_dir():
    """
    Ensures the Processed/ directory exists.
    Called once at startup by main.py.
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR, exist_ok=True)


def input_file_exists(path):
    """
    Returns True if the input file exists.
    """
    return os.path.exists(path)


def get_filename_from_path(path):
    """
    Extracts just the filename from a full UNC path.
    Example: '\\\\server\\folder\\file.wav' → 'file.wav'
    """
    return os.path.basename(path)


def build_output_paths(input_path):
    """
    Given an input file path, returns the expected output paths:
    - WAV output
    - transcript TXT
    - per-file LOG

    Example:
        input:  '\\server\\raw\\Alan\\AH_1987_04_12.wav'
        output: 'PipelineB/Processed/AH_1987_04_12.wav'
                'PipelineB/Processed/AH_1987_04_12.txt'
                'PipelineB/Processed/AH_1987_04_12.log'
    """
    filename = get_filename_from_path(input_path)

    wav_out = os.path.join(PROCESSED_DIR, filename)
    txt_out = os.path.join(PROCESSED_DIR, filename.replace(".wav", ".txt"))
    log_out = os.path.join(PROCESSED_DIR, filename.replace(".wav", ".log"))

    return wav_out, txt_out, log_out


def output_exists(input_path):
    """
    Returns True if the WAV output file already exists.
    This is the duplicate-protection step.
    """
    wav_out, _, _ = build_output_paths(input_path)
    return os.path.exists(wav_out)


def safe_write_binary(path, data):
    """
    Writes binary data (audio) safely.
    """
    with open(path, "wb") as f:
        f.write(data)


def safe_write_text(path, text):
    """
    Writes text data (transcript or logs) safely.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
