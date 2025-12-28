# audio_loader.py

import numpy as np
import soundfile as sf

from mutagen import File as MutagenFile
from mutagen.wave import WAVE
from mutagen.flac import FLAC
from mutagen.mp3 import MP3

from modules.logging_system import append_file_log
from modules.stepwise_saving import maybe_save_step_audio   # your existing helper


# ----------------------------------------------------------------------
# ORIGINAL FUNCTIONS (unchanged)
# ----------------------------------------------------------------------

def extract_embedded_metadata(path, log_buffer):
    """
    Extract embedded metadata from WAV/FLAC/MP3 using mutagen.
    Returns a dict of key/value pairs.
    """
    try:
        audio = MutagenFile(path, easy=True)

        if audio is None or audio.tags is None:
            append_file_log(log_buffer, "No embedded metadata found.")
            return {}

        md = {}

        # WAV (RIFF INFO tags)
        if isinstance(audio, WAVE):
            for k, v in audio.tags.items():
                md[k] = v[0] if isinstance(v, list) else v

        # FLAC Vorbis comments
        elif isinstance(audio, FLAC):
            for k, v in audio.tags.items():
                md[k] = v[0] if isinstance(v, list) else v

        # MP3 ID3 tags
        elif isinstance(audio, MP3):
            for k, v in audio.tags.items():
                try:
                    md[k] = v.text[0]
                except Exception:
                    pass

        append_file_log(log_buffer, f"Extracted embedded metadata: {md}")
        return md

    except Exception as e:
        append_file_log(log_buffer, f"Failed to extract embedded metadata: {e}")
        return {}


def load_audio(path, log_buffer, target_sr=None, dtype=np.float32):
    """
    Load audio from disk and return (waveform, sample_rate, embedded_metadata).
    """
    append_file_log(log_buffer, f"Loading audio from: {path}")

    # ---- Load audio samples ----
    try:
        waveform, sr = sf.read(path, dtype='float32')
        waveform = np.asarray(waveform, dtype=np.float32)
        append_file_log(log_buffer, f"Loaded with soundfile: sr={sr}, shape={waveform.shape}")
    except Exception as e:
        append_file_log(log_buffer, f"soundfile load failed: {e}")
        raise RuntimeError(f"Failed to load audio: {e}") from e

    # ---- Convert to mono ----
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
        append_file_log(log_buffer, "Converted to mono")

    # ---- Optional resample ----
    if target_sr is not None and sr != target_sr:
        try:
            import librosa
            append_file_log(log_buffer, f"Resampling from {sr} to {target_sr}")
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            append_file_log(log_buffer, f"Resample failed or librosa missing: {e}")

    # ---- Ensure dtype ----
    waveform = np.asarray(waveform, dtype=dtype, order='C')

    # ---- Extract metadata ----
    embedded_md = extract_embedded_metadata(path, log_buffer)

    append_file_log(
        log_buffer,
        f"Audio loaded successfully: sr={sr}, duration_s={len(waveform)/sr:.2f}, metadata_keys={list(embedded_md.keys())}"
    )

    return waveform, sr, embedded_md


# ----------------------------------------------------------------------
# NEW: STANDARDIZED DISPATCHER WRAPPER
# ----------------------------------------------------------------------

def run(state, ctx):
    """
    Standardized module entry point for the dispatcher.

    Dispatcher will call:
        audio_loader.run(state, ctx)

    This wrapper:
      - reads input_path from ctx
      - calls your existing load_audio()
      - updates state["waveform"], state["sr"], state["original_metadata"]
      - logs actions
      - triggers stepwise save (if enabled)
    """

    log_buffer = ctx["log_buffer"]
    input_path = ctx["working_copy_path"]   # this matches your existing main
    target_sr = ctx.get("target_sr")        # optional
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: load_audio ===")

    # Call your original function
    waveform, sr, embedded_md = load_audio(
        path=input_path,
        log_buffer=log_buffer,
        target_sr=target_sr
    )

    # Update state
    state["waveform"] = waveform
    state["sr"] = sr
    state["original_metadata"] = embedded_md

    # Record action
    state.setdefault("actions", []).append({
        "step": "load_audio",
        "time": time.time(),
        "sr": sr,
        "samples": len(waveform),
        "metadata_keys": list(embedded_md.keys())
    })

    # Stepwise save (dispatcher will not do this anymore)
    if save_stepwise:
        maybe_save_step_audio("load_audio", state, ctx)

    return state
