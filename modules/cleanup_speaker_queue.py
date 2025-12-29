#!/usr/bin/env python3
"""
modules/cleanup_speaker_queue.py

Dispatcher‑ready cleanup module for resolving unknown speakers and
correcting metadata in previously processed WAV files.

This module:
- Reads the speaker_resolution_queue.txt file
- Checks ONLY the last line for a RESOLVED entry
- If found, applies corrections to all files associated with that numeric speaker
- Rewrites embedded metadata + sidecar JSON
- Removes all processed lines from the queue
- Leaves only unresolved numeric speakers in the queue
- Safe for interruptions (queue is always authoritative)
"""

import os
import json
from mutagen.wave import WAVE
from modules.logging_system import append_file_log


QUEUE_PATH = "speaker_resolution_queue.txt"


# ---------------------------------------------------------
# QUEUE HELPERS
# ---------------------------------------------------------

def _read_queue():
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _write_queue(lines):
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _parse_resolved(line):
    # Example: RESOLVED speaker_00012 -> John Smith
    try:
        _, rest = line.split(" ", 1)
        numeric, real = rest.split("->")
        numeric = numeric.replace("RESOLVED", "").strip()
        real = real.strip()
        return numeric, real
    except Exception:
        return None, None


def _parse_unknown(line):
    # Example: UNKNOWN speaker_00012 | /path/to/file.wav
    try:
        _, rest = line.split(" ", 1)
        numeric, path = rest.split("|")
        numeric = numeric.replace("UNKNOWN", "").strip()
        path = path.strip()
        return numeric, path
    except Exception:
        return None, None


# ---------------------------------------------------------
# METADATA REWRITERS
# ---------------------------------------------------------

def _rewrite_wav_metadata(wav_path, speaker_name, log_buffer):
    """
    Replace embedded metadata speaker name in a WAV file.
    """
    try:
        audio = WAVE(wav_path)
        if audio.tags is None:
            audio.add_tags()

        audio["IART"] = speaker_name  # BSI uses IART for speaker/artist
        audio.save()
        append_file_log(log_buffer, f"cleanup: updated WAV metadata for {wav_path}")

    except Exception as e:
        append_file_log(log_buffer, f"cleanup_error: failed to update WAV metadata for {wav_path}: {e}")


def _rewrite_sidecar_json(wav_path, speaker_name, log_buffer):
    """
    Update the sidecar JSON file next to the WAV.
    """
    json_path = wav_path.replace(".wav", ".metadata.json")
    if not os.path.exists(json_path):
        append_file_log(log_buffer, f"cleanup: no sidecar JSON found for {wav_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "metadata" in data:
            data["metadata"]["speaker"] = speaker_name
        else:
            data["speaker"] = speaker_name

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        append_file_log(log_buffer, f"cleanup: updated sidecar JSON for {wav_path}")

    except Exception as e:
        append_file_log(log_buffer, f"cleanup_error: failed to update JSON for {wav_path}: {e}")


# ---------------------------------------------------------
# CORE CLEANUP LOGIC
# ---------------------------------------------------------

def cleanup_queue(log_buffer):
    """
    Performs cleanup if the last queue entry is RESOLVED.
    Called once per processed WAV file by the dispatcher.
    """
    lines = _read_queue()
    if not lines:
        append_file_log(log_buffer, "cleanup: queue empty, nothing to do")
        return

    last = lines[-1]

    if not last.startswith("RESOLVED"):
        append_file_log(log_buffer, "cleanup: no RESOLVED entry, skipping")
        return

    numeric, real_name = _parse_resolved(last)
    if not numeric or not real_name:
        append_file_log(log_buffer, "cleanup: malformed RESOLVED entry, skipping")
        return

    append_file_log(log_buffer, f"cleanup: resolving {numeric} -> {real_name}")

    # Find all UNKNOWN entries for this numeric speaker
    to_fix = []
    for line in lines:
        if line.startswith("UNKNOWN") and numeric in line:
            _, path = _parse_unknown(line)
            if path:
                to_fix.append(path)

    # Fix metadata for each file
    for wav_path in to_fix:
        _rewrite_wav_metadata(wav_path, real_name, log_buffer)
        _rewrite_sidecar_json(wav_path, real_name, log_buffer)

    # Remove all lines for this numeric speaker
    new_lines = [line for line in lines if numeric not in line]

    _write_queue(new_lines)

    append_file_log(log_buffer, f"cleanup: completed resolution for {numeric}")


# ---------------------------------------------------------
# DISPATCHER ENTRY POINT
# ---------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher‑compatible entry point.
    Does NOT modify state.
    """
    log_buffer = ctx["log_buffer"]
    append_file_log(log_buffer, "=== Step: cleanup_speaker_queue ===")

    cleanup_queue(log_buffer)

    # This step does not modify state
    return state
