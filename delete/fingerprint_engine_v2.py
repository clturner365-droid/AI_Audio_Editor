# modules/fingerprint_engine_v2.py
# Speaker fingerprint extraction and similarity utilities (v2).

import numpy as np
from numpy.linalg import norm
from modules.logging_system import append_file_log

try:
    import torch
    from pyannote.audio import Model
except ImportError:
    torch = None
    Model = None

_EMBED_MODEL = None
EMBED_MODEL_NAME = "pyannote/embedding"   # general-purpose speaker model
MIN_FP_NORM = 1e-6


def _load_embedding_model(log_buffer):
    """
    Loads the pyannote speaker embedding model once and caches it.
    Falls back to CPU if CUDA is unavailable.
    """
    global _EMBED_MODEL, Model, torch

    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    if Model is None or torch is None:
        append_file_log(log_buffer, "pyannote/torch not available; fingerprint extraction disabled.")
        return None

    append_file_log(log_buffer, "Loading pyannote speaker embedding model...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBED_MODEL = Model.from_pretrained(
            EMBED_MODEL_NAME,
            use_auth_token=None
        ).to(device)
        append_file_log(log_buffer, f"Speaker embedding model loaded on {device}.")
    except Exception as e:
        append_file_log(log_buffer, f"ERROR loading embedding model: {e}")
        _EMBED_MODEL = None

    return _EMBED_MODEL


def extract_fingerprint_v2(waveform: np.ndarray, sr: int, log_buffer):
    """
    Extracts a speaker embedding (voice fingerprint) using pyannote.audio.
    Returns:
        np.ndarray of shape (D,) or None on failure.
    """

    append_file_log(log_buffer, "v2: Extracting speaker fingerprint...")

    if waveform is None or len(waveform) == 0:
        append_file_log(log_buffer, "v2: Waveform empty; skipping fingerprint extraction.")
        return None

    waveform = np.asarray(waveform, dtype=np.float32).flatten()

    model = _load_embedding_model(log_buffer)
    if model is None or torch is None:
        append_file_log(log_buffer, "v2: Embedding model unavailable; returning None.")
        return None

    try:
        device = next(model.parameters()).device
        audio_tensor = torch.tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode():
            embedding = model(audio_tensor)

        embedding_np = embedding.detach().cpu().numpy().flatten()

        n = norm(embedding_np)
        if n < MIN_FP_NORM:
            append_file_log(log_buffer, "v2: Fingerprint norm too small; returning None.")
            return None

        embedding_np = embedding_np / n
        append_file_log(log_buffer, "v2: Fingerprint extracted successfully.")
        return embedding_np

    except Exception as e:
        append_file_log(log_buffer, f"v2: Fingerprint extraction failed: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    na = norm(a)
    nb = norm(b)
    if na < MIN_FP_NORM or nb < MIN_FP_NORM:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def confidence_from_similarity(sim: float) -> float:
    """
    Map cosine similarity [-1, 1] to [0, 1].
    """
    if sim < -1.0:
        sim = -1.0
    if sim > 1.0:
        sim = 1.0
    return (sim + 1.0) / 2.0
