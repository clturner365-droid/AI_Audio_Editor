#!/usr/bin/env python3
"""
modules/sermon_extraction.py

Transcript‑only sermon extraction.

This module:
- Uses ASR transcript segments to identify the sermon portion.
- Merges segments into blocks.
- Scores blocks using length + keyword density.
- Selects the best block (with minimum length requirement).
- Trims the audio to the sermon portion.
- Trims the transcript to match the sermon portion.
- Shifts transcript timestamps so the sermon starts at 0.0.
- Updates state["segments"], state["transcript"], and state["working_path"].
- Adds dispatcher wrapper: run(state, ctx)
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

from modules.logging_system import append_file_log
from modules.stepwise_saving import maybe_save_step_audio


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------

MIN_SERMON_LENGTH_S = 10 * 60  # 10 minutes
GAP_THRESHOLD_S = 2.0
KEYWORD_LIST = [
    "sermon", "scripture", "amen", "blessed", "gospel",
    "preach", "pray", "reading", "message"
]

LENGTH_WEIGHT = 0.50
KEYWORD_WEIGHT = 0.50   # now 100% transcript-based scoring


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _load_wave_and_sr(state: Dict[str, Any]):
    """Load audio from working_path or waveform+sr."""
    if state.get("working_path") and os.path.exists(state["working_path"]):
        wav, sr = sf.read(state["working_path"], dtype="float32")
        return wav, sr

    if state.get("waveform") is not None and state.get("sr") is not None:
        return np.asarray(state["waveform"], dtype="float32"), int(state["sr"])

    raise FileNotFoundError("No working_path or waveform in state")


def _merge_segments(segments: List[Dict[str, Any]], gap_thresh: float = GAP_THRESHOLD_S):
    """Merge adjacent transcript segments into larger blocks."""
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    merged = []

    cur = {
        "start": segs[0]["start"],
        "end": segs[0]["end"],
        "text": segs[0].get("text", "")
    }

    for s in segs[1:]:
        if s["start"] - cur["end"] <= gap_thresh:
            cur["end"] = s["end"]
            cur["text"] += " " + s.get("text", "")
        else:
            merged.append(cur)
            cur = {
                "start": s["start"],
                "end": s["end"],
                "text": s.get("text", "")
            }

    merged.append(cur)
    return merged


def _keyword_density(text: str) -> float:
    """Compute normalized keyword density 0..1."""
    if not text:
        return 0.0

    t = text.lower()
    count = sum(t.count(kw) for kw in KEYWORD_LIST)
    words = max(1, len(t.split()))
    density = (count / words) * 100.0
    return min(1.0, density / 2.0)


def _score_blocks(merged: List[Dict[str, Any]], wav_len_s: float):
    """Score merged blocks using length + keyword density."""
    scored = []

    for m in merged:
        start = float(m.get("start", 0.0))
        end = float(m.get("end", start))
        length = max(0.0, end - start)
        if length <= 0:
            continue

        length_score = min(1.0, length / max(1.0, wav_len_s))
        kd = _keyword_density(m.get("text", ""))

        score = (LENGTH_WEIGHT * length_score) + (KEYWORD_WEIGHT * kd)
        score = max(0.0, min(1.0, score))

        scored.append({
            "start": start,
            "end": end,
            "text": m.get("text", ""),
            "length_s": length,
            "length_score": length_score,
            "keyword_density": kd,
            "score": score
        })

    return sorted(scored, key=lambda b: b["score"], reverse=True)


def _write_wav_atomic(path: str, data: np.ndarray, sr: int):
    """Safe atomic write."""
    tmp = str(Path(path).with_suffix(".tmp"))
    sf.write(tmp, data, sr, subtype="PCM_16")
    os.replace(tmp, path)


def _trim_and_shift_transcript(segments, start_s, end_s):
    """Keep only transcript segments inside [start_s, end_s] and shift timestamps."""
    trimmed = []
    for seg in segments:
        s = seg.get("start", 0.0)
        e = seg.get("end", 0.0)

        if e <= start_s or s >= end_s:
            continue

        new_s = max(0.0, s - start_s)
        new_e = max(0.0, e - start_s)

        trimmed.append({
            "start": new_s,
            "end": new_e,
            "text": seg.get("text", "")
        })

    return trimmed


# ---------------------------------------------------------
# Main sermon extraction
# ---------------------------------------------------------

def remove_non_sermon_part(state: Dict[str, Any], log_buffer: list) -> Dict[str, Any]:
    """Transcript‑only sermon extraction."""
    segments = state.get("segments") or []
    if not segments:
        append_file_log(log_buffer, "sermon_extraction: no segments; skipping")
        state["sermon_selection"] = {
            "start_s": 0.0,
            "end_s": None,
            "score": 0.0,
            "reasons": ["no_segments"]
        }
        return state

    try:
        wav, sr = _load_wave_and_sr(state)
    except Exception as e:
        append_file_log(log_buffer, f"sermon_extraction: load error: {e}")
        state["sermon_selection"] = {
            "start_s": 0.0,
            "end_s": None,
            "score": 0.0,
            "reasons": ["load_error"]
        }
        return state

    total_duration = len(wav) / float(sr)
    merged = _merge_segments(segments)
    scored_blocks = _score_blocks(merged, total_duration)

    if not scored_blocks:
        append_file_log(log_buffer, "sermon_extraction: no scored blocks")
        state["sermon_selection"] = {
            "start_s": 0.0,
            "end_s": None,
            "score": 0.0,
            "reasons": ["no_scored_blocks"]
        }
        return state

    # Choose best block meeting minimum length
    chosen = next((b for b in scored_blocks if b["length_s"] >= MIN_SERMON_LENGTH_S), None)

    # If none meet min length, choose longest
    if chosen is None:
        chosen = max(scored_blocks, key=lambda x: x["length_s"])
        chosen["low_confidence_reason"] = "below_min_length"

    start_s = float(chosen["start"])
    end_s = float(chosen["end"])
    final_score = float(chosen["score"])

    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(wav), int(end_s * sr))

    if end_sample <= start_sample:
        append_file_log(log_buffer, "sermon_extraction: invalid boundaries")
        state["sermon_selection"] = {
            "start_s": 0.0,
            "end_s": None,
            "score": 0.0,
            "reasons": ["invalid_boundaries"]
        }
        return state

    # Trim audio
    base = Path(state.get("working_path", "audio")).stem
    out_dir = Path(state.get("working_path", ".")).parent
    sermon_path = str(out_dir / f"{base}-sermon.wav")

    try:
        trimmed = wav[start_sample:end_sample]
        _write_wav_atomic(sermon_path, trimmed, sr)

        state["working_path"] = sermon_path
        state.setdefault("final_paths", {})["sermon"] = sermon_path

        # Trim + shift transcript
        new_segments = _trim_and_shift_transcript(segments, start_s, end_s)
        state["segments"] = new_segments
        state["transcript"] = " ".join(seg["text"] for seg in new_segments)

        reasons = ["scored_selection"]
        if chosen.get("keyword_density", 0.0) > 0:
            reasons.append("keyword_density")
        if chosen.get("length_s", 0.0) < MIN_SERMON_LENGTH_S:
            reasons.append("below_min_length")

        state["sermon_selection"] = {
            "start_s": start_s,
            "end_s": end_s,
            "score": final_score,
            "reasons": reasons,
            "selected_block": chosen
        }

        append_file_log(log_buffer, f"sermon_extraction: wrote {sermon_path}")

    except Exception as e:
        append_file_log(log_buffer, f"sermon_extraction: write failed: {e}")
        state["sermon_selection"] = {
            "start_s": start_s,
            "end_s": end_s,
            "score": final_score,
            "reasons": ["write_failed"]
        }

    return state


# ---------------------------------------------------------
# Dispatcher wrapper
# ---------------------------------------------------------

def run(state, ctx):
    """Dispatcher entry point for sermon extraction."""
    log_buffer = ctx["log_buffer"]
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: sermon_extraction ===")

    try:
        state = remove_non_sermon_part(state, log_buffer)
    except Exception as e:
        append_file_log(log_buffer, f"sermon_extraction failed: {e}")
        return state

    state.setdefault("actions", []).append({
        "step": "sermon_extraction",
        "start_s": state.get("sermon_selection", {}).get("start_s"),
        "end_s": state.get("sermon_selection", {}).get("end_s"),
        "score": state.get("sermon_selection", {}).get("score")
    })

    if save_stepwise:
        maybe_save_step_audio("sermon_extraction", state, ctx)

    return state
