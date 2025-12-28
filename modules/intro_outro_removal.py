#!/usr/bin/env python3
"""
modules/intro_outro_removal.py

Function:
    remove_intros_outros(waveform, sr, log_buffer, *, segments=None, phrases=None,
                         max_search_s=300, similarity_threshold=0.78, padding_ms=300,
                         min_remaining_ratio=0.25)

Behavior:
- If `segments` (ASR time-aligned segments) is provided, use transcript-driven fuzzy matching
  to detect intro/outro phrases in the first/last `max_search_s` seconds.
- Otherwise, fall back to energy/music heuristics to find likely intro/outro regions.
- Never trim if the remaining audio would be shorter than `min_remaining_ratio` of original.
- Append a structured metadata entry to `log_buffer` under key "intro_outro_decision".
- Return the (possibly trimmed) waveform (numpy float32).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math
import json
import difflib
import time

from modules.logging_system import append_file_log
from modules.stepwise_saving import maybe_save_step_audio


# ----------------------------------------------------------------------
# Default phrase lists
# ----------------------------------------------------------------------

DEFAULT_OPENING_PHRASES = [
    "welcome", "good morning", "good evening", "welcome to", "we welcome", "let us welcome",
    "opening hymn", "opening prayer", "today we will", "today's message", "our scripture reading"
]

DEFAULT_CLOSING_PHRASES = [
    "amen", "closing hymn", "closing prayer", "benediction", "dismissal", "go in peace",
    "let us stand and sing", "final blessing", "may the lord bless you"
]

# Audio heuristic parameters
ENERGY_FRAME_MS = 200
SILENCE_ENERGY_THRESHOLD = 1e-6
MUSIC_ENERGY_RATIO_THRESHOLD = 0.6


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _join_early_text(segments: List[Dict[str, Any]], max_seconds: float) -> str:
    parts = []
    for s in segments:
        if s.get("start", 0.0) <= max_seconds:
            parts.append(s.get("text", ""))
        else:
            break
    return " ".join(parts).lower()


def _join_late_text(segments: List[Dict[str, Any]], max_seconds: float, total_duration: float) -> str:
    parts = []
    cutoff = max(0.0, total_duration - max_seconds)
    for s in segments:
        if s.get("end", 0.0) >= cutoff:
            parts.append(s.get("text", ""))
    return " ".join(parts).lower()


def _fuzzy_best_match(text: str, phrases: List[str]) -> Tuple[Optional[str], float]:
    if not text or not phrases:
        return None, 0.0
    best = None
    best_score = 0.0
    for p in phrases:
        score = difflib.SequenceMatcher(None, text, p.lower()).ratio()
        if score > best_score:
            best_score = score
            best = p
    return best, best_score


def _energy_frames(wav: np.ndarray, sr: int, frame_ms: int = ENERGY_FRAME_MS) -> Tuple[np.ndarray, int]:
    frame_len = max(1, int(sr * (frame_ms / 1000.0)))
    n_frames = math.ceil(len(wav) / frame_len)
    energies = np.zeros(n_frames, dtype=float)
    for i in range(n_frames):
        s = i * frame_len
        e = min(len(wav), s + frame_len)
        frame = wav[s:e]
        energies[i] = float(np.mean(frame * frame)) if frame.size else 0.0
    return energies, frame_len


def _find_leading_trailing_regions_by_energy(
    wav: np.ndarray,
    sr: int,
    padding_ms: int,
    energy_thresh: float
) -> Tuple[float, float]:

    energies, frame_len = _energy_frames(wav, sr)

    idx_first = 0
    while idx_first < len(energies) and energies[idx_first] <= energy_thresh:
        idx_first += 1

    idx_last = len(energies) - 1
    while idx_last >= 0 and energies[idx_last] <= energy_thresh:
        idx_last -= 1

    pad_samples = int(sr * (padding_ms / 1000.0))
    start_sample = max(0, idx_first * frame_len - pad_samples)
    end_sample = min(len(wav), (idx_last + 1) * frame_len + pad_samples)

    return start_sample / sr, end_sample / sr


# ----------------------------------------------------------------------
# Core trimming function
# ----------------------------------------------------------------------

def remove_intros_outros(
    waveform: np.ndarray,
    sr: int,
    log_buffer: List[str],
    *,
    segments: Optional[List[Dict[str, Any]]] = None,
    phrases: Optional[Dict[str, List[str]]] = None,
    max_search_s: float = 300.0,
    similarity_threshold: float = 0.78,
    padding_ms: int = 300,
    min_remaining_ratio: float = 0.25
) -> np.ndarray:

    try:
        total_samples = len(waveform)
        total_duration = total_samples / float(sr) if sr else 0.0

        phrases = phrases or {}
        opening_phrases = phrases.get("open", DEFAULT_OPENING_PHRASES)
        closing_phrases = phrases.get("close", DEFAULT_CLOSING_PHRASES)

        decision = {
            "method": None,
            "intro": None,
            "outro": None,
            "original_duration_s": total_duration,
            "trimmed_duration_s": None,
            "reasons": []
        }

        # --------------------------------------------------------------
        # Transcript-driven mode
        # --------------------------------------------------------------
        if segments:
            decision["method"] = "transcript_fuzzy"

            early_text = _join_early_text(segments, max_search_s)
            open_match, open_score = _fuzzy_best_match(early_text, opening_phrases)

            if open_match and open_score >= similarity_threshold:
                matched_start = 0.0
                matched_end = 0.0
                for seg in segments:
                    if seg.get("start", 0.0) > max_search_s:
                        break
                    if open_match.lower() in seg.get("text", "").lower():
                        matched_start = seg.get("start", 0.0)
                        matched_end = seg.get("end", matched_start)
                        break
                if matched_end == 0.0 and segments:
                    matched_end = segments[0].get("end", 0.0)

                trim_start_s = max(0.0, matched_end - (padding_ms / 1000.0))
                decision["intro"] = {
                    "phrase": open_match,
                    "similarity": open_score,
                    "trim_end_s": trim_start_s
                }
            else:
                decision["intro"] = {
                    "phrase": open_match,
                    "similarity": open_score,
                    "action": "no_match"
                }

            late_text = _join_late_text(segments, max_search_s, total_duration)
            close_match, close_score = _fuzzy_best_match(late_text, closing_phrases)

            if close_match and close_score >= similarity_threshold:
                matched_start = None
                matched_end = None
                for seg in reversed(segments):
                    if seg.get("end", 0.0) < (total_duration - max_search_s):
                        break
                    if close_match.lower() in seg.get("text", "").lower():
                        matched_start = seg.get("start", 0.0)
                        matched_end = seg.get("end", matched_start)
                        break
                if matched_start is None:
                    matched_start = segments[-1].get("start", max(0.0, total_duration - max_search_s))

                trim_end_s = min(total_duration, matched_start + (padding_ms / 1000.0))
                decision["outro"] = {
                    "phrase": close_match,
                    "similarity": close_score,
                    "trim_start_s": trim_end_s
                }
            else:
                decision["outro"] = {
                    "phrase": close_match,
                    "similarity": close_score,
                    "action": "no_match"
                }

            start_trim_s = decision["intro"].get("trim_end_s") or 0.0
            end_trim_s = decision["outro"].get("trim_start_s") or total_duration

            remaining = max(0.0, end_trim_s - start_trim_s)
            if total_duration > 0 and (remaining / total_duration) < min_remaining_ratio:
                decision["reasons"].append("remaining_too_short_after_transcript_trim; skipping trim")
                trimmed = waveform
            else:
                s_sample = int(start_trim_s * sr)
                e_sample = int(end_trim_s * sr)
                trimmed = waveform[s_sample:e_sample]
                decision["reasons"].append("transcript_trim_applied")

        # --------------------------------------------------------------
        # Energy-based heuristic mode
        # --------------------------------------------------------------
        else:
            decision["method"] = "audio_energy_heuristic"

            start_s, end_s = _find_leading_trailing_regions_by_energy(
                waveform, sr, padding_ms, SILENCE_ENERGY_THRESHOLD
            )

            if start_s <= 0.0 and end_s >= total_duration:
                decision["reasons"].append("no_energy_trim_needed")
                trimmed = waveform
            else:
                remaining = max(0.0, end_s - start_s)
                if total_duration > 0 and (remaining / total_duration) < min_remaining_ratio:
                    decision["reasons"].append("remaining_too_short_after_energy_trim; skipping trim")
                    trimmed = waveform
                else:
                    s_sample = int(start_s * sr)
                    e_sample = int(end_s * sr)
                    trimmed = waveform[s_sample:e_sample]
                    decision["intro"] = {"trim_end_s": start_s}
                    decision["outro"] = {"trim_start_s": end_s}
                    decision["reasons"].append("energy_trim_applied")

        decision["trimmed_duration_s"] = len(trimmed) / float(sr) if sr else None

        try:
            log_buffer.append("intro_outro_decision: " + json.dumps(decision))
        except Exception:
            log_buffer.append(f"intro_outro_decision: {decision}")

        return trimmed

    except Exception as e:
        try:
            log_buffer.append(f"intro_outro_error: {e}")
        except Exception:
            pass
        return waveform


# ----------------------------------------------------------------------
# Existing state-based API (unchanged)
# ----------------------------------------------------------------------

def remove_intros_outros_state(
    state: Dict[str, Any],
    log_buffer: List[str],
    *,
    phrases: Optional[Dict[str, List[str]]] = None,
    max_search_s: float = 300.0,
    similarity_threshold: float = 0.78,
    padding_ms: int = 300,
    min_remaining_ratio: float = 0.25
) -> Dict[str, Any]:

    wav = state.get("waveform")
    sr = state.get("sr")
    segments = state.get("segments")

    new_wav = remove_intros_outros(
        wav,
        sr,
        log_buffer,
        segments=segments,
        phrases=phrases,
        max_search_s=max_search_s,
        similarity_threshold=similarity_threshold,
        padding_ms=padding_ms,
        min_remaining_ratio=min_remaining_ratio
    )

    state["waveform"] = new_wav

    if log_buffer:
        last = log_buffer[-1]
        if isinstance(last, str) and last.startswith("intro_outro_decision: "):
            try:
                meta = json.loads(last.split("intro_outro_decision: ", 1)[1])
                state["intro_outro"] = meta
            except Exception:
                state.setdefault("intro_outro", {})["note"] = last

    return state


# ----------------------------------------------------------------------
# NEW: Dispatcher wrapper
# ----------------------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher entry point for intro/outro removal.
    """

    log_buffer = ctx["log_buffer"]
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: remove_intros_outros ===")

    state = remove_intros_outros_state(
        state,
        log_buffer,
        phrases=ctx.get("intro_outro_phrases"),
        max_search_s=ctx.get("intro_outro_max_search_s", 300.0),
        similarity_threshold=ctx.get("intro_outro_similarity_threshold", 0.78),
        padding_ms=ctx.get("intro_outro_padding_ms", 300),
        min_remaining_ratio=ctx.get("intro_outro_min_remaining_ratio", 0.25),
    )

    state.setdefault("actions", []).append({
        "step": "remove_intros_outros",
        "time": time.time(),
        "result": state.get("intro_outro", {})
    })

    if save_stepwise:
        maybe_save_step_audio("remove_intros_outros", state, ctx)

    return state
