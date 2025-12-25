# modules/loudness_normalization.py
# Real loudness normalization using pyloudnorm (EBU R128 standard).

import numpy as np
import pyloudnorm as pyln
import librosa
from modules.logging_system import append_file_log


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def normalize_loudness(waveform: np.ndarray, sr: int, target_lufs: float, log_buffer):
    """
    Normalize audio to a target LUFS level using EBU R128.
    Steps:
      1. Measure current LUFS
      2. Compute required gain
      3. Apply gain safely (no clipping)
      4. Log everything
    """

    append_file_log(log_buffer, f"Normalizing loudness to {target_lufs} LUFS...")

    if len(waveform) == 0:
        append_file_log(log_buffer, "Waveform empty; skipping loudness normalization.")
        return waveform

    # Ensure float32 mono
    waveform = waveform.astype(np.float32).flatten()

    # Create loudness meter
    meter = pyln.Meter(sr)  # EBU R128 meter

    try:
        # 1. Measure current loudness
        current_lufs = meter.integrated_loudness(waveform)
        append_file_log(log_buffer, f"Current loudness: {current_lufs:.2f} LUFS")

        # 2. Compute required gain
        gain_db = target_lufs - current_lufs
        append_file_log(log_buffer, f"Applying gain: {gain_db:.2f} dB")

        # Convert dB to linear gain
        gain = 10 ** (gain_db / 20)

        # 3. Apply gain
        normalized = waveform * gain

        # 4. Safety: prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.999:
            append_file_log(log_buffer, f"Peak {peak:.3f} detected; applying limiter.")
            normalized = normalized / peak * 0.98  # soft limiter

        # 5. Re-measure for logging
        final_lufs = meter.integrated_loudness(normalized)
        append_file_log(log_buffer, f"Final loudness: {final_lufs:.2f} LUFS")

        return normalized.astype(np.float32)

    except Exception as e:
        append_file_log(log_buffer, f"Loudness normalization failed: {e}")
        return waveform
