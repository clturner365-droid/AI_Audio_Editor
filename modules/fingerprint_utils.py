# modules/fingerprint_utils.py
# Speaker fingerprinting using pyannote.audio embeddings.

import numpy as np
import torch
from pyannote.audio import Model
from pyannote.audio.pipelines.utils.hook import ProgressHook
from modules.logging_system import append_file_log


# ---------------------------------------------------------
# GLOBAL MODEL (loaded once)
# ---------------------------------------------------------

_EMBED_MODEL = None
EMBED_MODEL_NAME = "pyannote/embedding"   # best general-purpose speaker model


def _load_embedding_model(log_buffer):
    """
    Loads the pyannote speaker embedding model once and caches it.
    """
    global _EMBED_MODEL

    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    append_file_log(log_buffer, "Loading pyannote speaker embedding model...")

    try:
        _EMBED_MODEL = Model.from_pretrained(
            EMBED_MODEL_NAME,
            use_auth_token=None  # not needed for this model
        ).to("cuda")

        append_file_log(log_buffer, "Speaker embedding model loaded on GPU.")

    except Exception as e:
        append_file_log(log_buffer, f"ERROR loading embedding model: {e}")
        _EMBED_MODEL = None

    return _EMBED_MODEL


# ---------------------------------------------------------
# MAIN FINGERPRINT EXTRACTION
# ---------------------------------------------------------

def extract_fingerprint(waveform: np.ndarray, sr: int, log_buffer):
    """
    Extracts a speaker embedding (voice fingerprint) using pyannote.audio.
    Returns:
        np.ndarray of shape (512,) or None on failure.
    """

    append_file_log(log_buffer, "Extracting speaker fingerprint...")

    if len(waveform) == 0:
        append_file_log(log_buffer, "Waveform empty; skipping fingerprint extraction.")
        return None

    # Ensure float32 mono
    waveform = waveform.astype(np.float32).flatten()

    # Load model (cached)
    model = _load_embedding_model(log_buffer)
    if model is None:
        append_file_log(log_buffer, "Embedding model unavailable; returning None.")
        return None

    try:
        # Convert to torch tensor
        audio_tensor = torch.tensor(waveform).float().to("cuda")

        # Pyannote expects shape (batch, time)
        audio_tensor = audio_tensor.unsqueeze(0)

        # Extract embedding
        with torch.inference_mode():
            embedding = model(audio_tensor)

        # Convert to numpy
        embedding_np = embedding.cpu().numpy().flatten()

        append_file_log(log_buffer, "Fingerprint extracted successfully.")

        return embedding_np

    except Exception as e:
        append_file_log(log_buffer, f"Fingerprint extraction failed: {e}")
        return None
