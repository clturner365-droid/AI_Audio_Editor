import os
from TTS.api import TTS
from pydub import AudioSegment
import tempfile

# Cache the model so it loads only once
_tts_model = None

def _load_tts_model():
    global _tts_model
    if _tts_model is None:
        # You can change this model to any Coqui TTS model you prefer
        _tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    return _tts_model


def generate_tts_audio(text, config=None, log_buffer=None):
    """
    Generates spoken audio from text using Coqui TTS.
    Returns a pydub.AudioSegment.
    """

    if log_buffer is not None:
        log_buffer.append(f"Coqui TTS: generating audio for text: {text}")

    tts = _load_tts_model()

    # Create a temporary WAV file for TTS output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name

    # Generate speech to the temp file
    tts.tts_to_file(text=text, file_path=temp_path)

    # Load into pydub
    audio = AudioSegment.from_wav(temp_path)

    # Clean up temp file
    try:
        os.remove(temp_path)
    except:
        pass

    if log_buffer is not None:
        log_buffer.append("Coqui TTS: audio generation complete")

    return audio

