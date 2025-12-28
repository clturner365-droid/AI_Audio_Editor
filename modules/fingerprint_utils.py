# modules/fingerprint_utils.py
# Speaker fingerprinting using pyannote.audio embeddings.

import numpy as np
import torch
from numpy.linalg import norm
from pyannote.audio import Model
from pyannote.audio.pipelines.utils.hook import ProgressHook
import time

from modules.logging_system import append_file_log
from modules.stepwise_saving import maybe_save_step_audio


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
            use_auth_token=None
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

    waveform = waveform.astype(np.float32).flatten()

    model = _load_embedding_model(log_buffer)
    if model is None:
        append_file_log(log_buffer, "Embedding model unavailable; returning None.")
        return None

    try:
        audio_tensor = torch.tensor(waveform).float().to("cuda")
        audio_tensor = audio_tensor.unsqueeze(0)

        with torch.inference_mode():
            embedding = model(audio_tensor)

        embedding_np = embedding.cpu().numpy().flatten()

        append_file_log(log_buffer, "Fingerprint extracted successfully.")

        return embedding_np

    except Exception as e:
        append_file_log(log_buffer, f"Fingerprint extraction failed: {e}")
        return None


# ---------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return -1.0
    if norm(a) == 0 or norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# ---------------------------------------------------------
# COMPARE FINGERPRINTS
# ---------------------------------------------------------

def compare_fingerprints(registry, canonical_names, fp_new, log_buffer):
    """
    Compare new fingerprint to stored fingerprints.
    Returns:
        (best_match_name, confidence_score)
    """

    append_file_log(log_buffer, "Comparing fingerprint to registry...")

    if fp_new is None:
        append_file_log(log_buffer, "New fingerprint is None; cannot compare.")
        return None, 0.0

    best_match = None
    best_score = -1.0

    for name in canonical_names:
        stored_fp = registry.get("fingerprints", {}).get(name)

        if stored_fp is None:
            append_file_log(log_buffer, f"No stored fingerprint for {name}.")
            continue

        stored_fp = np.array(stored_fp, dtype=np.float32)

        score = _cosine_similarity(fp_new, stored_fp)
        append_file_log(log_buffer, f"Similarity with {name}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_match = name

    confidence = max(0.0, min(1.0, (best_score + 1) / 2))

    append_file_log(
        log_buffer,
        f"Best match: {best_match} (confidence={confidence:.2f})"
    )

    return best_match, confidence


# ---------------------------------------------------------
# DISPATCHER WRAPPER
# ---------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher entry point for fingerprint extraction.

    This wrapper:
      - pulls waveform and sr from state
      - calls extract_fingerprint()
      - stores embedding in state["fingerprint"]
      - logs the action
      - triggers stepwise save if enabled
    """

    log_buffer = ctx["log_buffer"]
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: extract_fingerprint ===")

    wav = state.get("waveform")
    sr = state.get("sr")

    if wav is None or sr is None:
        append_file_log(log_buffer, "No waveform or sample rate in state; skipping fingerprint extraction.")
        return state

    fp = extract_fingerprint(wav, sr, log_buffer)
    state["fingerprint"] = fp

    state.setdefault("actions", []).append({
        "step": "extract_fingerprint",
        "time": time.time(),
        "success": fp is not None
    })

    if save_stepwise:
        maybe_save_step_audio("extract_fingerprint", state, ctx)

    return state
