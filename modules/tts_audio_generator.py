#!/usr/bin/env python3
"""
modules/tts_audio_generator.py

Generates TTS audio using Coqui TTS with rotating male VCTK voices.
Returns a pydub.AudioSegment and does NOT modify pipeline state.
"""

import os
import tempfile
from pathlib import Path

from TTS.api import TTS
from pydub import AudioSegment

from modules.logging_system import append_file_log


# ---------------------------------------------------------
# Male VCTK voices (clean, natural, stable)
# ---------------------------------------------------------

MALE_VOICES = ["p225", "p233", "p236", "p245", "p248"]


# ---------------------------------------------------------
# Model cache
# ---------------------------------------------------------

_tts_model = None


def _load_tts_model():
    """
    Load and cache the VCTK VITS model.
    This model supports multiple speakers, including the male voices we rotate through.
    """
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS("tts_models/en/vctk/vits")
    return _tts_model


# ---------------------------------------------------------
# Main TTS function
# ---------------------------------------------------------

def generate_tts_audio(text: str, *, file_index: int = 0, log_buffer=None) -> AudioSegment:
    """
    Generate spoken audio from text using rotating male VCTK voices.
    - text: the text to speak
    - file_index: determines which male voice to use (rotates automatically)
    Returns a pydub.AudioSegment.
    """

    if log_buffer is not None:
        append_file_log(log_buffer, f"TTS: generating audio for text: {text}")

    try:
        tts = _load_tts_model()

        # Pick voice based on rotation
        speaker = MALE_VOICES[file_index % len(MALE_VOICES)]

        if log_buffer is not None:
            append_file_log(log_buffer, f"TTS: using speaker '{speaker}'")

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
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if log_buffer is not None:
        append_file_log(log_buffer, "TTS: audio generation complete")

    return audio
