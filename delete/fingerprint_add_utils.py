# modules/fingerprint_utils.py
# (continuation) Update stored fingerprint safely.

import numpy as np
import json
from modules.logging_system import append_file_log

# Thresholds
UPDATE_CONFIDENCE_THRESHOLD = 0.75   # only update if confidence >= this
MIN_FP_NORM = 1e-6                   # avoid zero vectors


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    if v is None:
        return None
    v = np.array(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < MIN_FP_NORM:
        return None
    return v / n


def update_fingerprint_if_safe(registry: dict, canonical_names: list, fp_new: np.ndarray,
                               confidence: float, multi: bool, log_buffer):
    """
    Decide whether to update the stored fingerprint for the best match.
    Behavior:
      - If `multi` is True (multi-speaker file), do not update.
      - If confidence < threshold, do not update.
      - If no stored fingerprint exists for the matched name, store fp_new if confidence high.
      - If stored fingerprint exists, update by weighted average (simple running average).
    Returns:
      updated_registry (dict), updated_name (str or None), action (str)
    """

    append_file_log(log_buffer, "Evaluating whether to update fingerprint...")

    if fp_new is None:
        append_file_log(log_buffer, "New fingerprint is None; skipping update.")
        return registry, None, "no_fp"

    if multi:
        append_file_log(log_buffer, "File marked multi-speaker; skipping fingerprint update.")
        return registry, None, "multi_skip"

    if confidence < UPDATE_CONFIDENCE_THRESHOLD:
        append_file_log(log_buffer, f"Confidence {confidence:.2f} below threshold; skipping update.")
        return registry, None, "low_confidence"

    # Choose best match name by comparing to registry (reuse compare_fingerprints externally)
    # Here we assume caller already determined best_match; if not, we fallback to first canonical name.
    best_match = None
    if len(canonical_names) > 0:
        best_match = canonical_names[0]

    # If registry has no fingerprints key, create it
    if "fingerprints" not in registry:
        registry["fingerprints"] = {}

    # Normalize new fingerprint
    fp_new_n = _normalize_vec(fp_new)
    if fp_new_n is None:
        append_file_log(log_buffer, "Normalized new fingerprint invalid; skipping update.")
        return registry, None, "invalid_fp"

    # If no stored fingerprint for best_match, store it
    stored = registry["fingerprints"].get(best_match)
    if stored is None:
        registry["fingerprints"][best_match] = fp_new_n.tolist()
        append_file_log(log_buffer, f"Stored new fingerprint for {best_match}.")
        return registry, best_match, "stored_new"

    # Otherwise, compute running average: new = normalize(alpha * stored + (1-alpha) * fp_new)
    try:
        stored_np = np.array(stored, dtype=np.float32)
        alpha = 0.6  # weight existing fingerprint more to avoid drift
        updated = alpha * stored_np + (1.0 - alpha) * fp_new_n
        updated_n = _normalize_vec(updated)
        if updated_n is None:
            append_file_log(log_buffer, "Updated fingerprint invalid after normalization; skipping update.")
            return registry, None, "update_failed"

        registry["fingerprints"][best_match] = updated_n.tolist()
        append_file_log(log_buffer, f"Updated fingerprint for {best_match} (alpha={alpha}).")
        return registry, best_match, "updated"

    except Exception as e:
        append_file_log(log_buffer, f"Fingerprint update failed: {e}")
        return registry, None, "error"
