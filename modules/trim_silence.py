# modules/silence_trim.py
# Removes leading and trailing silence using librosa.effects.trim.

import numpy as np
import librosa
from modules.logging_system import append_file_log


# Silence threshold in dB
TRIM_TOP_DB = 40   # Higher = more aggressive trimming


def trim_silence(waveform: np.ndarray, sr: int, log_buffer):
    """
    Removes leading and trailing silence using librosa's trim function.
    This is a safe, speech-friendly method that avoids cutting words.
    """

    append_file_log(log_buffer, "Trimming leading/trailing silence...")

    if len(waveform) == 0:
        append_file_log(log_buffer, "Waveform empty; skipping silence trim.")
        return waveform

    # Ensure float32 mono
    waveform = waveform.astype(np.float32).flatten()

    try:
        trimmed, idx = librosa.effects.trim(
            waveform,
            top_db=TRIM_TOP_DB
        )

        start_sample, end_sample = idx
        start_sec = start_sample / sr
        end_sec = (len(waveform) - end_sample) / sr

        append_file_log(
            log_buffer,
            f"Silence trimmed: start={start_sec:.2f}s, end={end_sec:.2f}s."
        )

        return trimmed.astype(np.float32)

    except Exception as e:
        append_file_log(log_buffer, f"Silence trim failed: {e}")
        return waveform
