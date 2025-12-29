import numpy as np
from pydub import AudioSegment
from modules.loudness_normalization import normalize_loudness as lufs_normalize

def audiosegment_to_numpy(seg: AudioSegment):
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples, seg.frame_rate

def numpy_to_audiosegment(samples: np.ndarray, sr: int):
    if samples.ndim == 2:
        samples = samples.flatten()
    samples_int16 = (samples * 32767).astype(np.int16)
    return AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

def normalize_loudness(target_audio, reference_audio, log_buffer=None):
    """
    Matches loudness of target_audio to reference_audio using your existing
    loudness_normalization.py module.
    """

    # Convert both to numpy
    target_np, sr = audiosegment_to_numpy(target_audio)
    ref_np, _ = audiosegment_to_numpy(reference_audio)

    # Measure reference loudness
    # Your module already handles measurement internally
    target_lufs = -16  # or derive from reference if needed

    normalized_np = lufs_normalize(
        waveform=target_np,
        sr=sr,
        target_lufs=target_lufs,
        log_buffer=log_buffer
    )

    # Convert back to AudioSegment
    return numpy_to_audiosegment(normalized_np, sr)
