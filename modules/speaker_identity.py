# modules/speaker_identity.py
# Unified speaker identity + fingerprint update step.

import os
import numpy as np
import time

from modules.logging_system import append_file_log
from modules.fingerprint_engine_v2 import cosine_similarity, confidence_from_similarity
from modules.stepwise_saving import maybe_save_step_audio


QUEUE_PATH = "speaker_resolution_queue.txt"
AMBIGUOUS_PATH = "ambiguous_speakers.txt"
NEW_SPEAKERS_PATH = "new_speakers.txt"

NUMERIC_PREFIX = "speaker_"
MATCH_THRESHOLD = 0.75
UPDATE_THRESHOLD = 0.75
ALPHA_EXISTING = 0.6


# ---------------------------------------------------------
# File helpers
# ---------------------------------------------------------

def _append_line(path: str, line: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
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


def append_ambiguous(original_name: str, numeric_or_name: str, wav_path: str, log_buffer):
    line = f"{original_name} | {numeric_or_name} | {wav_path}"
    _append_line(AMBIGUOUS_PATH, line)
    append_file_log(log_buffer, f"v2: ambiguous -> {line}")


def append_new_speaker(original_name: str, log_buffer):
    if not original_name:
        return
    line = original_name.strip()
    if not line:
        return
    _append_line(NEW_SPEAKERS_PATH, line)
    append_file_log(log_buffer, f"v2: new speaker name discovered -> {line}")


# ---------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------

def is_numeric_id(name: str) -> bool:
    return isinstance(name, str) and name.startswith(NUMERIC_PREFIX)


def allocate_numeric_id(registry: dict) -> str:
    existing = registry.get("fingerprints", {})
    max_idx = 0
    for name in existing.keys():
        if is_numeric_id(name):
            try:
                idx = int(name.replace(NUMERIC_PREFIX, ""))
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return f"{NUMERIC_PREFIX}{max_idx + 1:05d}"


# ---------------------------------------------------------
# Identity + matching logic
# ---------------------------------------------------------

def determine_identity_and_match(
    registry: dict,
    canonical_names: list,
    fp_new: np.ndarray,
    original_meta_name: str,
    input_path: str,
    log_buffer
):
    append_file_log(log_buffer, "v2: Determining speaker identity...")

    if fp_new is None:
        append_file_log(log_buffer, "v2: No fingerprint; cannot determine identity.")
        return None, None, 0.0

    registry.setdefault("fingerprints", {})
    fp_store = registry["fingerprints"]

    original_meta_name = (original_meta_name or "").strip()
    expected_name = None

    # Metadata handling
    if original_meta_name and original_meta_name in canonical_names:
        expected_name = original_meta_name
        append_file_log(log_buffer, f"v2: Expected speaker from metadata: {expected_name}")
    elif original_meta_name:
        append_file_log(log_buffer, f"v2: Metadata name not in canonical list: {original_meta_name}")
        append_new_speaker(original_meta_name, log_buffer)
        append_file_log(log_buffer, f"v2: Metadata name '{original_meta_name}' ignored for identity matching.")

    best_match_name = None
    best_conf = 0.0

    # 1) Direct expected-name check
    if expected_name and expected_name in fp_store:
        try:
            stored_np = np.array(fp_store[expected_name], dtype=np.float32)
            sim = cosine_similarity(fp_new, stored_np)
            conf = confidence_from_similarity(sim)
            append_file_log(log_buffer, f"v2: Similarity with expected {expected_name}: {sim:.4f}, conf={conf:.2f}")
            best_match_name = expected_name
            best_conf = conf
        except Exception as e:
            append_file_log(log_buffer, f"v2: error comparing to expected {expected_name}: {e}")

    if expected_name and best_conf >= MATCH_THRESHOLD:
        append_file_log(log_buffer, f"v2: Accepted expected speaker {expected_name} with confidence {best_conf:.2f}")
        return expected_name, expected_name, best_conf

    if expected_name and best_conf < MATCH_THRESHOLD:
        append_file_log(
            log_buffer,
            f"v2: Expected speaker {expected_name} rejected due to low confidence "
            f"({best_conf:.2f} < {MATCH_THRESHOLD})."
        )

    if expected_name and expected_name not in fp_store:
        append_file_log(log_buffer, f"v2: First time for known speaker {expected_name}; enrolling.")
        return expected_name, None, 0.0

    # 2) RESOLVED logic: numeric ID → real name
    if original_meta_name in canonical_names:
        real_name = original_meta_name

        for name, stored_fp in fp_store.items():
            if not is_numeric_id(name):
                continue

            try:
                stored_np = np.array(stored_fp, dtype=np.float32)
                sim = cosine_similarity(fp_new, stored_np)
                conf = confidence_from_similarity(sim)
            except Exception as e:
                append_file_log(log_buffer, f"v2: error comparing to numeric ID {name}: {e}")
                continue

            if conf < MATCH_THRESHOLD:
                append_file_log(
                    log_buffer,
                    f"v2: Numeric ID {name} NOT promoted to real name {real_name} "
                    f"due to low confidence ({conf:.2f} < {MATCH_THRESHOLD})."
                )
                continue

            append_file_log(
                log_buffer,
                f"v2: Numeric ID {name} matches real speaker {real_name} (conf={conf:.2f}); promoting."
            )

            append_queue_resolved(name, real_name, log_buffer)

            fp_store[real_name] = stored_fp
            del fp_store[name]

            return real_name, real_name, conf

    # 3) Registry-wide matching
    best_global_name = None
    best_global_conf = 0.0

    for name, stored_fp in fp_store.items():
        try:
            stored_np = np.array(stored_fp, dtype=np.float32)
            sim = cosine_similarity(fp_new, stored_np)
            conf = confidence_from_similarity(sim)
            append_file_log(log_buffer, f"v2: registry-wide match check: {name} sim={sim:.4f} conf={conf:.2f}")
            if conf > best_global_conf:
                best_global_conf = conf
                best_global_name = name
        except Exception as e:
            append_file_log(log_buffer, f"v2: error comparing to {name}: {e}")

    append_file_log(
        log_buffer,
        f"v2: registry-wide summary: best={best_global_name}, conf={best_global_conf:.2f}"
    )

    if best_global_conf < MATCH_THRESHOLD:
        append_file_log(
            log_buffer,
            f"v2: No registry-wide match above threshold ({best_global_conf:.2f} < {MATCH_THRESHOLD})."
        )
    else:
        append_file_log(
            log_buffer,
            f"v2: registry-wide best match: {best_global_name} (conf={best_global_conf:.2f})"
        )

        if is_numeric_id(best_global_name) and original_meta_name in canonical_names:
            real_name = original_meta_name
            append_file_log(
                log_buffer,
                f"v2: numeric ID {best_global_name} matches real name {real_name}; promoting."
            )
            append_queue_resolved(best_global_name, real_name, log_buffer)
            fp_store[real_name] = fp_store[best_global_name]
            del fp_store[best_global_name]
            return real_name, real_name, best_global_conf

        if best_global_name in canonical_names:
            if original_meta_name and original_meta_name != best_global_name:
                append_ambiguous(original_meta_name, best_global_name, input_path, log_buffer)
            return best_global_name, best_global_name, best_global_conf

        if is_numeric_id(best_global_name):
            return best_global_name, best_global_name, best_global_conf

    # 4) Fallback: assign numeric ID
    numeric_id = allocate_numeric_id(registry)

    append_file_log(
        log_buffer,
        "v2: Falling back to numeric ID because: "
        f"expected_name={expected_name}, "
        f"best_conf={best_conf:.2f}, "
        f"best_global_name={best_global_name}, "
        f"best_global_conf={best_global_conf:.2f}"
    )
    append_file_log(log_buffer, f"v2: Assigning numeric ID {numeric_id}.")

    append_queue_unknown(numeric_id, input_path, log_buffer)

    if original_meta_name:
        append_ambiguous(original_meta_name, numeric_id, input_path, log_buffer)

    return numeric_id, None, 0.0


# ---------------------------------------------------------
# Fingerprint update logic
# ---------------------------------------------------------

def update_fingerprint_for_decided_speaker(
    registry: dict,
    decided_name: str,
    fp_new: np.ndarray,
    confidence: float,
    multi: bool,
    log_buffer
):
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
        fp_store[decided_name] = fp_new.tolist()
        append_file_log(log_buffer, f"v2: Stored new fingerprint for {decided_name}.")
        return registry, "stored_new"

    if confidence < UPDATE_THRESHOLD:
        append_file_log(
            log_buffer,
            f"v2: confidence {confidence:.2f} below UPDATE_THRESHOLD; skipping update."
        )
        return registry, "low_confidence"

    try:
        existing = np.array(stored_fp, dtype=np.float32)
        updated = ALPHA_EXISTING * existing + (1.0 - ALPHA_EXISTING) * fp_new
        n = np.linalg.norm(updated)
        if n < 1e-6:
            append_file_log(log_buffer, "v2: updated fingerprint norm too small; skipping update.")
            return registry, "invalid_update"

        fp_store[decided_name] = (updated / n).tolist()
        append_file_log(log_buffer, f"v2: Updated fingerprint for {decided_name} (alpha={ALPHA_EXISTING}).")
        return registry, "updated"

    except Exception as e:
        append_file_log(log_buffer, f"v2: fingerprint update failed for {decided_name}: {e}")
        return registry, "error"


# ---------------------------------------------------------
# Unified Dispatcher Wrapper — speaker_identity
# ---------------------------------------------------------

def run(state, ctx):
    """
    Unified dispatcher step:
      1. Determine identity
      2. Update fingerprint
      3. Update registry
      4. Log everything
    """

    log_buffer = ctx["log_buffer"]
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: speaker_identity ===")

    registry = state.get("registry", {})
    fp_new = state.get("fingerprint")
    canonical_names = ctx.get("canonical_speakers", [])
    original_meta_name = state.get("metadata_speaker", "")
    input_path = state.get("input_path", "")
    multi = bool(state.get("multi_speaker", False))

    # 1) Determine identity
    decided_name, resolved_name, confidence = determine_identity_and_match(
        registry,
        canonical_names,
        fp_new,
        original_meta_name,
        input_path,
        log_buffer
    )

    # 2) Update fingerprint
    registry, update_result = update_fingerprint_for_decided_speaker(
        registry,
        decided_name,
        fp_new,
        confidence,
        multi,
        log_buffer
    )

    # 3) Update state
    state["decided_speaker"] = decided_name
    state["resolved_speaker"] = resolved_name
    state["identity_confidence"] = confidence
    state["update_result"] = update_result
    state["registry"] = registry

    # 4) Log action
    state.setdefault("actions", []).append({
        "step": "speaker_identity",
        "time": time.time(),
        "decided": decided_name,
        "resolved": resolved_name,
        "confidence": confidence,
        "update_result": update_result
    })

    # 5) Stepwise save
    if save_stepwise:
        maybe_save_step_audio("speaker_identity", state, ctx)

    return state
