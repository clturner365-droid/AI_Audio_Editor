# modules/silence_trim.py
# Removes leading and trailing silence using librosa.effects.trim.

import numpy as np
import librosa
import time

from modules.logging_system import append_file_log
from modules.stepwise_saving import maybe_save_step_audio


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


# ----------------------------------------------------------------------
# Dispatcher wrapper
# ----------------------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher entry point for silence trimming.

    This wrapper:
      - pulls waveform and sr from state
      - calls your existing trim_silence()
      - updates state["waveform"]
      - logs the action
      - triggers stepwise save if enabled
    """

    log_buffer = ctx["log_buffer"]
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: trim_silence ===")

    wav = state.get("waveform")
    sr = state.get("sr")

    if wav is None or sr is None:
        append_file_log(log_buffer, "No waveform or sample rate in state; skipping silence trim.")
        return state

    trimmed = trim_silence(wav, sr, log_buffer)
    state["waveform"] = trimmed

    # Record action
    state.setdefault("actions", []).append({
        "step": "trim_silence",
        "time": time.time(),
        "original_samples": len(wav),
        "trimmed_samples": len(trimmed)
    })

    # Stepwise save
    if save_stepwise:
        maybe_save_step_audio("trim_silence", state, ctx)

    return state
