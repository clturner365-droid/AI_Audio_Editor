# fingerprint.py
# Handles fingerprint extraction, comparison, and update logic.

import numpy as np
from modules.logging_system import append_file_log, log_fingerprint_error
from modules.registry import update_fingerprint


# ---------------------------------------------------------
# FINGERPRINT EXTRACTION
# ---------------------------------------------------------

def extract_fingerprint(waveform, sample_rate, log_buffer):
    """
    Extracts a fingerprint vector from the audio.
    Placeholder for real ML model.
    Returns a numpy array representing the fingerprint.
    """

    append_file_log(log_buffer, "Extracting fingerprint...")

    # TODO: Replace with real fingerprint model
    # Placeholder: random vector
    fp = np.random.rand(128).astype(np.float32)

    return fp


# ---------------------------------------------------------
# FINGERPRINT COMPARISON
# ---------------------------------------------------------

def fingerprint_distance(fp1, fp2):
    """
    Computes a distance metric between two fingerprint vectors.
    Lower = more similar.
    """
    if fp1 is None or fp2 is None:
        return None

    # Cosine distance
    dot = np.dot(fp1, fp2)
    norm = np.linalg.norm(fp1) * np.linalg.norm(fp2)
    if norm == 0:
        return None

    similarity = dot / norm
    distance = 1 - similarity
    return distance


def compute_confidence(distance):
    """
    Converts distance into a confidence score.
    Placeholder logic.
    """
    if distance is None:
        return 0.0

    # Example: invert distance
    confidence = max(0.0, 1.0 - distance)
    return confidence


# ---------------------------------------------------------
# MATCHING LOGIC
# ---------------------------------------------------------

def match_speaker_fingerprint(registry, canonical_name, fp_new, log_buffer):
    """
    Compares the new fingerprint to the stored one.
    Returns:
        - match: True/False
        - confidence: float
    """

    entry = registry.get(canonical_name)
    if not entry:
        append_file_log(log_buffer, f"No registry entry for {canonical_name}.")
        return False, 0.0

    fp_old = entry.get("fingerprint")

    if fp_old is None:
        append_file_log(log_buffer, f"No existing fingerprint for {canonical_name}.")
        return False, 0.0

    distance = fingerprint_distance(fp_old, fp_new)
    confidence = compute_confidence(distance)

    append_file_log(log_buffer, f"Fingerprint match confidence: {confidence:.3f}")

    return confidence >= 0.75, confidence  # threshold placeholder


# ---------------------------------------------------------
# MULTI-SPEAKER SAFETY
# ---------------------------------------------------------

def safe_update_fingerprint(registry, canonical_name, fp_new, confidence, multi, log_buffer):
    """
    Updates fingerprint ONLY if:
        - single speaker
        - confidence is high enough
    """

    if multi:
        append_file_log(log_buffer, "MULTI-SPEAKER: Skipping fingerprint update.")
        return

    if confidence < 0.60:
        append_file_log(log_buffer, "Low confidence: fingerprint NOT updated.")
        return

    update_fingerprint(registry, canonical_name, fp_new.tolist(), confidence)
    append_file_log(log_buffer, "Fingerprint updated successfully.")


# ---------------------------------------------------------
# FULL FINGERPRINT PIPELINE
# ---------------------------------------------------------

def process_fingerprint(registry, canonical_names, waveform, sample_rate, multi, log_buffer):
    """
    Full fingerprint workflow:
        1. Extract fingerprint
        2. If multi-speaker → verification only
        3. If single-speaker → match + update
    """

    fp_new = extract_fingerprint(waveform, sample_rate, log_buffer)

    if multi:
        append_file_log(log_buffer, "MULTI-SPEAKER: Fingerprint verification only.")
        return fp_new, None

    # Single speaker → match & update
    canonical = canonical_names[0]

    match, confidence = match_speaker_fingerprint(registry, canonical, fp_new, log_buffer)

    if not match:
        append_file_log(log_buffer, "Fingerprint mismatch detected.")
        log_fingerprint_error(f"Mismatch for speaker {canonical}")
    else:
        append_file_log(log_buffer, "Fingerprint match confirmed.")

    safe_update_fingerprint(registry, canonical, fp_new, confidence, multi, log_buffer)

    return fp_new, confidence
