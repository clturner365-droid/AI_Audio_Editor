#!/usr/bin/env python3
"""
modules/tts_audio_generator.py

Generates TTS audio using Coqui TTS (VCTK VITS).
- Exposes a list of recommended male voices (MALE_VOICES).
- Caller chooses the speaker and passes it in.
- Returns a pydub.AudioSegment and does NOT modify pipeline state.
"""

import os
import tempfile

from TTS.api import TTS
from pydub import AudioSegment

from modules.logging_system import append_file_log


# ---------------------------------------------------------
# Recommended male VCTK voices (clean, natural, stable)
# ---------------------------------------------------------

MALE_VOICES = ["p225", "p233", "p236", "p245", "p248"]


# ---------------------------------------------------------
# Model cache
# ---------------------------------------------------------

_tts_model = None


def _load_tts_model():
    """
    Load and cache the VCTK VITS model.
    This model supports multiple speakers, including the male voices we use.
    """
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS("tts_models/en/vctk/vits")
    return _tts_model


# ---------------------------------------------------------
# Main TTS function
# ---------------------------------------------------------

def generate_tts_audio(text: str, *, speaker: str, log_buffer=None) -> AudioSegment:
    """
    Generate spoken audio from text using a specific VCTK speaker.

    Args:
        text: Text to synthesize.
        speaker: VCTK speaker ID (e.g., "p225", "p233", ...).
        log_buffer: Pipeline log buffer.

    Returns:
        pydub.AudioSegment with the spoken audio.
    """

    if log_buffer is not None:
        append_file_log(log_buffer, f"TTS: generating audio for text: {text}")
        append_file_log(log_buffer, f"TTS: using speaker '{speaker}'")

    temp_path = None

    try:
        tts = _load_tts_model()

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name

        # Generate speech into the temp file
        tts.tts_to_file(text=text, speaker=speaker, file_path=temp_path)

        # Load into pydub
        audio = AudioSegment.from_wav(temp_path)

    except Exception as e:
        if log_buffer is not None:
            append_file_log(log_buffer, f"TTS: generation failed: {e}")
        raise

    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                # Non-fatal; temp file will be cleaned by OS eventually
                pass

    if log_buffer is not None:
        append_file_log(log_buffer, "TTS: audio generation complete")

    return audio
