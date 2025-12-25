# (continuation of fingerprint_utils.py)

import numpy as np
from numpy.linalg import norm
from modules.logging_system import append_file_log


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

    # Loop through all canonical speakers
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

    # Confidence scaling
    confidence = max(0.0, min(1.0, (best_score + 1) / 2))

    append_file_log(
        log_buffer,
        f"Best match: {best_match} (confidence={confidence:.2f})"
    )

    return best_match, confidence
