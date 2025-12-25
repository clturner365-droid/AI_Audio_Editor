# audio_loader.py
# Loads audio from disk and returns waveform + sample rate.
# Currently a placeholder until real DSP libraries are installed.

import numpy as np
from modules.logging_system import append_file_log

# If using librosa or pydub later, you will import them here:
# import librosa
# from pydub import AudioSegment


def load_audio(path, log_buffer):
    """
    Loads WAV audio from disk.
    Placeholder implementation returns a dummy waveform.
    Replace with real audio loading when DSP libraries are available.
    """
    append_file_log(log_buffer, f"Loading audio from: {path}")

    # TODO: Replace with real audio loading
    # Example (future):
    # waveform, sr = librosa.load(path, sr=None, mono=True)

    # Placeholder: 1 second of silence at 16 kHz
    sample_rate = 16000
    waveform = np.zeros(sample_rate, dtype=np.float32)

    append_file_log(log_buffer, "Audio loaded (placeholder).")

    return waveform, sample_rate
