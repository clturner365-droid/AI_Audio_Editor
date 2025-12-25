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

# default phrase lists (extendable)
DEFAULT_OPENING_PHRASES = [
    "welcome", "good morning", "good evening", "welcome to", "we welcome", "let us welcome",
    "opening hymn", "opening prayer", "today we will", "today's message", "our scripture reading"
]
DEFAULT_CLOSING_PHRASES = [
    "amen", "closing hymn", "closing prayer", "benediction", "dismissal", "go in peace",
    "let us stand and sing", "final blessing", "may the lord bless you"
]

# audio heuristics parameters
ENERGY_FRAME_MS = 200
SILENCE_ENERGY_THRESHOLD = 1e-6  # tuned for float32; adjust if needed
MUSIC_ENERGY_RATIO_THRESHOLD = 0.6  # if music energy dominates, treat as non-sermon


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
        # token-based similarity via SequenceMatcher on lowercased strings
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


def _find_leading_trailing_regions_by_energy(wav: np.ndarray, sr: int, padding_ms: int, energy_thresh: float) -> Tuple[float, float]:
    energies, frame_len = _energy_frames(wav, sr)
    # find first frame above threshold
    idx_first = 0
    while idx_first < len(energies) and energies[idx_first] <= energy_thresh:
        idx_first += 1
    idx_last = len(energies) - 1
    while idx_last >= 0 and energies[idx_last] <= energy_thresh:
        idx_last -= 1
    # convert to seconds with padding
    pad_samples = int(sr * (padding_ms / 1000.0))
    start_sample = max(0, idx_first * frame_len - pad_samples)
    end_sample = min(len(wav), (idx_last + 1) * frame_len + pad_samples)
    start_s = start_sample / sr
    end_s = end_sample / sr
    return start_s, end_s


def remove_intros_outros(waveform: np.ndarray, sr: int, log_buffer: List[str], *,
                         segments: Optional[List[Dict[str, Any]]] = None,
                         phrases: Optional[Dict[str, List[str]]] = None,
                         max_search_s: float = 300.0,
                         similarity_threshold: float = 0.78,
                         padding_ms: int = 300,
                         min_remaining_ratio: float = 0.25) -> np.ndarray:
    """
    Trim intro/outro from waveform. Returns trimmed waveform (numpy float32).
    - segments: optional ASR segments (list of dicts with start,end,text)
    - phrases: optional dict {"open": [...], "close": [...]} to override defaults
    """
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

        # Transcript-driven mode
        if segments:
            decision["method"] = "transcript_fuzzy"
            # early text
            early_text = _join_early_text(segments, max_search_s)
            open_match, open_score = _fuzzy_best_match(early_text, opening_phrases)
            if open_match and open_score >= similarity_threshold:
                # find earliest segment that contains the matched phrase (best-effort)
                matched_start = 0.0
                matched_end = 0.0
                # search segments for phrase substring
                for seg in segments:
                    if seg.get("start", 0.0) > max_search_s:
                        break
                    if open_match.lower() in seg.get("text", "").lower():
                        matched_start = seg.get("start", 0.0)
                        matched_end = seg.get("end", matched_start)
                        break
                # fallback: if not found, trim up to first non-empty segment end
                if matched_end == 0.0 and segments:
                    matched_end = segments[0].get("end", 0.0)
                # apply padding
                trim_start_s = max(0.0, matched_end - (padding_ms / 1000.0))
                decision["intro"] = {"phrase": open_match, "similarity": open_score, "trim_end_s": trim_start_s}
            else:
                decision["intro"] = {"phrase": open_match, "similarity": open_score, "action": "no_match"}

            # late text
            late_text = _join_late_text(segments, max_search_s, total_duration)
            close_match, close_score = _fuzzy_best_match(late_text, closing_phrases)
            if close_match and close_score >= similarity_threshold:
                # find last segment containing phrase
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
                    # fallback: use last segment start
                    matched_start = segments[-1].get("start", max(0.0, total_duration - max_search_s))
                trim_end_s = min(total_duration, matched_start + (padding_ms / 1000.0))
                decision["outro"] = {"phrase": close_match, "similarity": close_score, "trim_start_s": trim_end_s}
            else:
                decision["outro"] = {"phrase": close_match, "similarity": close_score, "action": "no_match"}

            # compute final trim boundaries
            start_trim_s = decision["intro"].get("trim_end_s") if decision["intro"].get("trim_end_s") is not None else 0.0
            end_trim_s = decision["outro"].get("trim_start_s") if decision["outro"].get("trim_start_s") is not None else total_duration

            # safety: ensure we don't trim too much
            remaining = max(0.0, end_trim_s - start_trim_s)
            if total_duration > 0 and (remaining / total_duration) < min_remaining_ratio:
                decision["reasons"].append("remaining_too_short_after_transcript_trim; skipping trim")
                trimmed = waveform
            else:
                s_sample = int(start_trim_s * sr)
                e_sample = int(end_trim_s * sr)
                trimmed = waveform[s_sample:e_sample]
                decision["reasons"].append("transcript_trim_applied")
        else:
            # Audio heuristic mode
            decision["method"] = "audio_energy_heuristic"
            # compute energy frames and find leading/trailing low-energy regions
            start_s, end_s = _find_leading_trailing_regions_by_energy(waveform, sr, padding_ms, SILENCE_ENERGY_THRESHOLD)
            # safety: if energy heuristic returns full file, skip
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
        # append structured metadata to log_buffer for downstream modules to parse
        try:
            log_buffer.append("intro_outro_decision: " + json.dumps(decision))
        except Exception:
            # fallback to human-readable
            log_buffer.append(f"intro_outro_decision: {decision}")

        return trimmed
    except Exception as e:
        # on error, log and return original waveform
        try:
            log_buffer.append(f"intro_outro_error: {e}")
        except Exception:
            pass
        return waveform


# Optional state-based API for future integration (not used by current main.py)
def remove_intros_outros_state(state: Dict[str, Any], log_buffer: List[str], *,
                                phrases: Optional[Dict[str, List[str]]] = None,
                                max_search_s: float = 300.0,
                                similarity_threshold: float = 0.78,
                                padding_ms: int = 300,
                                min_remaining_ratio: float = 0.25) -> Dict[str, Any]:
    """
    Accepts and returns a state dict. Uses state['waveform'], state['sr'], and optional state['segments'].
    Updates state with 'intro_outro' metadata and sets state['waveform'] to the trimmed waveform.
    """
    wav = state.get("waveform")
    sr = state.get("sr")
    segments = state.get("segments")
    new_wav = remove_intros_outros(wav, sr, log_buffer, segments=segments, phrases=phrases,
                                   max_search_s=max_search_s, similarity_threshold=similarity_threshold,
                                   padding_ms=padding_ms, min_remaining_ratio=min_remaining_ratio)
    state["waveform"] = new_wav
    # parse last log entry if present to populate structured metadata
    if log_buffer:
        last = log_buffer[-1]
        if isinstance(last, str) and last.startswith("intro_outro_decision: "):
            try:
                meta = json.loads(last.split("intro_outro_decision: ", 1)[1])
                state["intro_outro"] = meta
            except Exception:
                state.setdefault("intro_outro", {})["note"] = last
    return state
