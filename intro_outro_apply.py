#!/usr/bin/env python3
"""
modules/intro_outro_apply.py

Applies stored intro/outro audio to a cleaned sermon waveform.

Enhancements:
- Idempotent: attempts to detect if the intro or outro is already present and skips reapplication.
- Safe concatenation with short crossfade to avoid clicks.
- Resamples templates to target sample rate and ensures mono float32.
- Optional state update when a `state` dict is provided (non-breaking).
- Detailed logging via append_file_log.
"""

from typing import Optional, Dict, Any
import numpy as np
import librosa
from modules.logging_system import append_file_log

# Helper to load an intro/outro file and resample to target sr
def _load_template(path: str, target_sr: int, log_buffer):
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        append_file_log(log_buffer, f"Loaded template {path}: {len(y)} samples @ {target_sr} Hz.")
        return y.astype(np.float32)
    except Exception as e:
        append_file_log(log_buffer, f"Failed to load template {path}: {e}")
        return None


def _rms(arr: np.ndarray) -> float:
    if arr is None or arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _normalize(arr: np.ndarray) -> np.ndarray:
    maxv = np.max(np.abs(arr)) if arr.size else 1.0
    if maxv <= 0:
        return arr
    return arr / maxv


def _cross_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute a simple normalized cross-correlation similarity between two 1-D arrays.
    Returns a value in 0..1 where higher means more similar.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    # align lengths
    n = min(len(a), len(b))
    a_seg = a[:n].astype(np.float64)
    b_seg = b[:n].astype(np.float64)
    # normalize
    a_seg = a_seg - np.mean(a_seg)
    b_seg = b_seg - np.mean(b_seg)
    denom = (np.linalg.norm(a_seg) * np.linalg.norm(b_seg)) + 1e-12
    corr = float(np.dot(a_seg, b_seg) / denom)
    # map from -1..1 to 0..1
    return max(0.0, min(1.0, (corr + 1.0) / 2.0))


def _apply_crossfade(prefix: np.ndarray, body: np.ndarray, fade_ms: int, sr: int) -> np.ndarray:
    """
    Crossfade prefix into the start of body over fade_ms milliseconds.
    If prefix is longer than fade region, only the tail of prefix is crossfaded.
    """
    if fade_ms <= 0 or sr <= 0:
        return np.concatenate([prefix, body])
    fade_samples = int(sr * (fade_ms / 1000.0))
    if fade_samples <= 0:
        return np.concatenate([prefix, body])
    # ensure arrays are float32
    prefix = prefix.astype(np.float32)
    body = body.astype(np.float32)
    # determine overlap slices
    tail_prefix = prefix[-fade_samples:] if len(prefix) >= fade_samples else np.pad(prefix, (fade_samples - len(prefix), 0))
    head_body = body[:fade_samples] if len(body) >= fade_samples else np.pad(body, (0, fade_samples - len(body)))
    # create fades
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = 1.0 - fade_in
    cross = (tail_prefix * fade_out) + (head_body * fade_in)
    # assemble final
    pre_part = prefix[:-fade_samples] if len(prefix) > fade_samples else np.array([], dtype=np.float32)
    post_part = body[fade_samples:] if len(body) > fade_samples else np.array([], dtype=np.float32)
    return np.concatenate([pre_part, cross, post_part])


def _apply_tail_crossfade(body: np.ndarray, suffix: np.ndarray, fade_ms: int, sr: int) -> np.ndarray:
    """
    Crossfade suffix into the end of body over fade_ms milliseconds.
    """
    if fade_ms <= 0 or sr <= 0:
        return np.concatenate([body, suffix])
    fade_samples = int(sr * (fade_ms / 1000.0))
    if fade_samples <= 0:
        return np.concatenate([body, suffix])
    body = body.astype(np.float32)
    suffix = suffix.astype(np.float32)
    tail_body = body[-fade_samples:] if len(body) >= fade_samples else np.pad(body, (fade_samples - len(body), 0))
    head_suffix = suffix[:fade_samples] if len(suffix) >= fade_samples else np.pad(suffix, (0, fade_samples - len(suffix)))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = 1.0 - fade_in
    cross = (tail_body * fade_out) + (head_suffix * fade_in)
    pre_part = body[:-fade_samples] if len(body) > fade_samples else np.array([], dtype=np.float32)
    post_part = suffix[fade_samples:] if len(suffix) > fade_samples else np.array([], dtype=np.float32)
    return np.concatenate([pre_part, cross, post_part])


def apply_intro_outro(cleaned_waveform: np.ndarray, sr: int, registry: Dict[str, Any], log_buffer, *,
                      state: Optional[Dict[str, Any]] = None,
                      fade_ms: int = 150,
                      similarity_threshold: float = 0.85,
                      allow_duplicate_check: bool = True) -> np.ndarray:
    """
    If registry contains 'intro_path' or 'outro_path', load them and concatenate:
      final = intro + cleaned + outro
    Idempotency:
      - If allow_duplicate_check is True, the function attempts to detect whether the intro
        or outro is already present (via short cross-similarity) and will skip reapplying.
    Optional:
      - If a `state` dict is provided, the function will set state['intro_outro_applied'] = True
        when templates are applied (non-breaking).
    Returns the final waveform (numpy float32). Does not write files or modify originals.
    """

    append_file_log(log_buffer, "Applying intro/outro templates if available...")

    if cleaned_waveform is None or len(cleaned_waveform) == 0:
        append_file_log(log_buffer, "Cleaned waveform empty; skipping apply.")
        return cleaned_waveform

    intro = None
    outro = None

    intro_path = registry.get("intro_path")
    outro_path = registry.get("outro_path")

    if intro_path:
        intro = _load_template(intro_path, sr, log_buffer)
    if outro_path:
        outro = _load_template(outro_path, sr, log_buffer)

    # If nothing to apply, return early
    if intro is None and outro is None:
        append_file_log(log_buffer, "No intro/outro templates available; returning cleaned waveform unchanged.")
        return cleaned_waveform

    # Idempotency checks: try to detect if intro/outro already present
    applied_intro = False
    applied_outro = False
    try:
        if allow_duplicate_check and intro is not None:
            # compare intro to start of cleaned waveform
            n = min(len(intro), len(cleaned_waveform))
            sim = _cross_similarity(_normalize(intro[:n]), _normalize(cleaned_waveform[:n]))
            append_file_log(log_buffer, f"Intro similarity to start