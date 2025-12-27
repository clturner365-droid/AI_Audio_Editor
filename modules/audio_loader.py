# audio_loader.py

import numpy as np
import soundfile as sf

from mutagen import File as MutagenFile
from mutagen.wave import WAVE
from mutagen.flac import FLAC
from mutagen.mp3 import MP3

from modules.logging_system import append_file_log


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

    - Uses soundfile for primary load (preserves original SR and channels).
    - Converts to mono.
    - Optional resampling (if target_sr is provided).
    - Always returns numpy.float32.
    - Extracts embedded metadata using mutagen.
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

