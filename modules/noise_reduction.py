# modules/noise_reduction.py
# Real noise reduction using noisereduce + optional VAD smoothing.

import numpy as np
import noisereduce as nr
import webrtcvad
import librosa
from modules.logging_system import append_file_log


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

VAD_FRAME_MS = 30               # 10, 20, or 30 ms allowed
VAD_AGGRESSIVENESS = 2          # 0–3 (higher = more aggressive)
NOISE_REDUCTION_PROP = 0.8      # 0.0–1.0 (how much noise to remove)
MIN_SPEECH_SECONDS = 5          # safety check


# ---------------------------------------------------------
# HELPER: WebRTC VAD mask
# ---------------------------------------------------------

def _vad_mask(waveform, sr, log_buffer):
    """
    Returns a boolean mask of where speech is detected.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    frame_len = int(sr * (VAD_FRAME_MS / 1000.0))
    num_frames = len(waveform) // frame_len

    mask = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        frame = waveform[start:end]

        # Convert to 16‑bit PCM for VAD
        pcm = (frame * 32768).astype(np.int16).tobytes()

        try:
            mask[i] = vad.is_speech(pcm, sr)
        except:
            mask[i] = True  # fail‑safe: assume speech

    append_file_log(log_buffer, f"VAD: {mask.sum()} speech frames out of {len(mask)}.")

    # Expand mask back to sample resolution
    expanded = np.repeat(mask, frame_len)
    expanded = expanded[:len(waveform)]

    return expanded


# ---------------------------------------------------------
# MAIN NOISE REDUCTION FUNCTION
# ---------------------------------------------------------

def reduce_noise(waveform, sr, log_buffer):
    """
    Performs noise reduction using:
      - WebRTC VAD to identify speech regions
      - noisereduce spectral gating
      - safety checks to avoid over‑processing
    """

    append_file_log(log_buffer, "Starting noise reduction...")

    if len(waveform) == 0:
        append_file_log(log_buffer, "Waveform empty; skipping noise reduction.")
        return waveform

    # Safety: ensure float32 mono
    waveform = waveform.astype(np.float32).flatten()

    # -----------------------------------------------------
    # 1. Compute VAD mask (speech vs noise)
    # -----------------------------------------------------
    vad_mask = _vad_mask(waveform, sr, log_buffer)

    speech_ratio = vad_mask.mean()
    append_file_log(log_buffer, f"Speech ratio: {speech_ratio:.2f}")

    # If almost all speech, skip heavy noise reduction
    if speech_ratio > 0.95:
        append_file_log(log_buffer, "Audio is mostly speech; skipping noise reduction.")
        return waveform

    # -----------------------------------------------------
    # 2. Estimate noise profile from non‑speech regions
    # -----------------------------------------------------
    noise_profile = waveform[~vad_mask]

    if len(noise_profile) < sr:  # less than 1 second of noise
        append_file_log(log_buffer, "Not enough noise-only audio; using global reduction.")
        noise_profile = waveform[:sr]  # fallback: first second

    # -----------------------------------------------------
    # 3. Apply spectral noise reduction
    # -----------------------------------------------------
    try:
        cleaned = nr.reduce_noise(
            y=waveform,
            y_noise=noise_profile,
            prop_decrease=NOISE_REDUCTION_PROP,
            verbose=False
        )
        append_file_log(log_buffer, "Spectral noise reduction applied.")

    except Exception as e:
        append_file_log(log_buffer, f"Noise reduction failed: {e}")
        return waveform

    # -----------------------------------------------------
    # 4. Safety: ensure we didn’t destroy the audio
    # -----------------------------------------------------
    cleaned_energy = np.sqrt(np.mean(cleaned ** 2))
    original_energy = np.sqrt(np.mean(waveform ** 2))

    if cleaned_energy < original_energy * 0.25:
        append_file_log(log_buffer, "Noise reduction too aggressive; reverting to original.")
        return waveform

    # -----------------------------------------------------
    # 5. Optional smoothing (reduce artifacts)
    # -----------------------------------------------------
    cleaned = librosa.effects.preemphasis(cleaned, coef=0.97)
    cleaned = librosa.effects.deemphasis(cleaned, coef=0.97)

    append_file_log(log_buffer, "Noise reduction complete.")

    return cleaned.astype(np.float32)
