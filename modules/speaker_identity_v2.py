# modules/speaker_identity_v2.py
# Speaker identity decision logic, numeric-ID handling, and queue/log writing.

import os
import numpy as np
from modules.logging_system import append_file_log
from modules.fingerprint_engine_v2 import cosine_similarity, confidence_from_similarity

QUEUE_PATH = "speaker_resolution_queue.txt"
AMBIGUOUS_PATH = "ambiguous_speakers.txt"
NEW_SPEAKERS_PATH = "new_speakers.txt"

NUMERIC_PREFIX = "speaker_"
MATCH_THRESHOLD = 0.75        # confidence >= this = accept
UPDATE_THRESHOLD = 0.75       # required to update fingerprint
ALPHA_EXISTING = 0.6          # running-average weight


# ------------------------ file helpers ------------------------

def _append_line(path: str, line: str):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_queue_unknown(numeric_id: str, wav_path: str, log_buffer):
    line = f"UNKNOWN {numeric_id} | {wav_path}"
    _append_line(QUEUE_PATH, line)
    append_file_log(log_buffer, f"v2: queue UNKNOWN -> {line}")


def append_queue_resolved(numeric_id: str, real_name: str, log_buffer):
    line = f"RESOLVED {numeric_id} -> {real_name}"
    _append_line(QUEUE_PATH, line)
    append_file_log(log_buffer, f"v2: queue RESOLVED -> {line}")


def append_ambiguous(original_name: str, numeric_id: str, wav_path: str, log_buffer):
    line = f"{original_name} | {numeric_id} | {wav_path}"
    _append_line(AMBIGUOUS_PATH, line)
    append_file_log(log_buffer, f"v2: ambiguous -> {line}")


def append_new_speaker(original_name: str, log_buffer):
    if not original_name:
        return
    line = original_name.strip()
    _append_line(NEW_SPEAKERS_PATH, line)
    append_file_log(log_buffer, f"v2: new speaker name discovered -> {line}")


# ------------------------ naming helpers ------------------------

def is_numeric_id(name: str) -> bool:
    return isinstance(name, str) and name.startswith(NUMERIC_PREFIX)


def allocate_numeric_id(registry: dict) -> str:
    """
    Find the next available numeric speaker ID based on registry["fingerprints"] keys.
    """
    existing = registry.get("fingerprints", {})
    max_idx = 0
    for name in existing.keys():
        if is_numeric_id(name):
            try:
                idx = int(name.replace(NUMERIC_PREFIX, ""))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    next_idx = max_idx + 1
    return f"{NUMERIC_PREFIX}{next_idx:05d}"


# ------------------------ core logic ------------------------

def determine_identity_and_match(
    registry: dict,
    canonical_names: list,
    fp_new: np.ndarray,
    original_meta_name: str,
    input_path: str,
    log_buffer
):
    """
    Decide which speaker this file belongs to, based on:
      - canonical name from metadata (if known)
      - existing fingerprints in registry
      - new fingerprint (fp_new)

    Returns:
      decided_name (str or None),
      best_match_name (str or None),
      confidence (float)
    """

    append_file_log(log_buffer, "v2: Determining speaker identity...")

    if fp_new is None:
        append_file_log(log_buffer, "v2: No fingerprint; cannot determine identity.")
        return None, None, 0.0

    # Ensure registry["fingerprints"] exists
    registry.setdefault("fingerprints", {})
    fp_store = registry["fingerprints"]

    original_meta_name = (original_meta_name or "").strip()
    expected_name = None

    # If metadata name is in canonical list, treat it as the expected canonical name
    if original_meta_name and original_meta_name in canonical_names:
        expected_name = original_meta_name
        append_file_log(log_buffer, f"v2: Expected speaker from metadata: {expected_name}")
    elif original_meta_name:
        # Metadata has a name not in canonical list -> new real speaker candidate
        append_file_log(log_buffer, f"v2: Metadata name not in canonical list: {original_meta_name}")
        append_new_speaker(original_meta_name, log_buffer)

    # 1) If we have an expected name with a stored fingerprint, compare directly
    best_match_name = None
    best_conf = 0.0

    if expected_name and expected_name in fp_store:
        sim = cosine_similarity(fp_new, np.array(fp_store[expected_name], dtype=np.float32))
        conf = confidence_from_similarity(sim)
        append_file_log(log_buffer, f"v2: Similarity with expected {expected_name}: {sim:.4f}, conf={conf:.2f}")
        best_match_name = expected_name
        best_conf = conf

    # 2) If no good match yet, we *could* scan all registry fingerprints.
    # For now, keep it simple: rely on expected_name only.
    # (You can expand here later to search all speakers.)

    # Decision:
    if expected_name and best_conf >= MATCH_THRESHOLD:
        # Trusted match to expected speaker
        decided = expected_name
        append_file_log(log_buffer, f"v2: Accepted expected speaker {decided} with confidence {best_conf:.2f}")
        return decided, best_match_name, best_conf

    if expected_name and expected_name not in fp_store:
        # First time seeing this known speaker -> enroll under expected_name
        decided = expected_name
        append_file_log(log_buffer, f"v2: First time for known speaker {decided}; will enroll fingerprint.")
        return decided, None, 0.0

    # At this point, we do NOT trust any mapping to an existing named speaker.
    # Treat this as unknown -> assign numeric ID.
    numeric_id = allocate_numeric_id(registry)
    append_file_log(log_buffer, f"v2: No reliable match; assigning numeric ID {numeric_id}.")

    # Queue entry for cleanup
    append_queue_unknown(numeric_id, input_path, log_buffer)

    # Ambiguous record if we had a metadata name
    if original_meta_name:
        append_ambiguous(original_meta_name, numeric_id, input_path, log_buffer)

    decided = numeric_id
    return decided, None, 0.0


def update_fingerprint_for_decided_speaker(
    registry: dict,
    decided_name: str,
    fp_new: np.ndarray,
    confidence: float,
    multi: bool,
    log_buffer
):
    """
    Update or create fingerprint for the decided speaker identity.
    Behavior:
      - If multi is True, do NOT update fingerprints.
      - If no stored fingerprint for decided_name, store fp_new.
      - If stored fingerprint exists AND confidence >= UPDATE_THRESHOLD,
        do a running average update.
    Returns:
      registry (possibly modified), action (str)
    """

    append_file_log(log_buffer, f"v2: Updating fingerprint for decided speaker {decided_name}...")

    if fp_new is None:
        append_file_log(log_buffer, "v2: fp_new is None; skipping update.")
        return registry, "no_fp"

    if multi:
        append_file_log(log_buffer, "v2: multi-speaker file; skipping fingerprint update.")
        return registry, "multi_skip"

    registry.setdefault("fingerprints", {})
    fp_store = registry["fingerprints"]

    if decided_name is None:
        append_file_log(log_buffer, "v2: decided_name is None; skipping update.")
        return registry, "no_name"

    stored_fp = fp_store.get(decided_name)

    if stored_fp is None:
        # No fingerprint yet -> simple enrollment
        fp_store[decided_name] = fp_new.tolist()
        append_file_log(log_buffer, f"v2: Stored new fingerprint for {decided_name}.")
        return registry, "stored_new"

    # We have an existing fingerprint; update only if confidence high enough
    if confidence < UPDATE_THRESHOLD:
        append_file_log(log_buffer, f"v2: confidence {confidence:.2f} below UPDATE_THRESHOLD; skipping update.")
        return registry, "low_confidence"

    try:
        existing = np.array(stored_fp, dtype=np.float32)
        updated = ALPHA_EXISTING * existing + (1.0 - ALPHA_EXISTING) * fp_new
        n = np.linalg.norm(updated)
        if n < 1e-6:
            append_file_log(log_buffer, "v2: updated fingerprint norm too small; skipping update.")
            return registry, "invalid_update"

        updated = (updated / n).tolist()
        fp_store[decided_name] = updated
        append_file_log(log_buffer, f"v2: Updated fingerprint for {decided_name} (alpha={ALPHA_EXISTING}).")
        return registry, "updated"

    except Exception as e:
        append_file_log(log_buffer, f"v2: fingerprint update failed for {decided_name}: {e}")
        return registry, "error"
