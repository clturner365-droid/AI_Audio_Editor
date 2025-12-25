#!/usr/bin/env python3
"""
modules/sermon_extraction.py

Provides `remove_non_sermon_part(state, log_buffer, enrollment_registry, device, test_mode)`.

Enhancements:
- Scores candidate blocks using length, keyword density, and optional speaker verification.
- Selects best contiguous block(s) and enforces configurable minimum sermon length.
- Writes a trimmed sermon WAV next to working_path without overwriting originals.
- Populates state["sermon_selection"] with detailed scoring and reasons.
- Safe behavior when diarization/pyannote or enrollment registry are absent.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf

# Optional pyannote imports for embeddings (if available)
try:
    from pyannote.audio import Inference as PyannoteInference  # type: ignore
except Exception:
    PyannoteInference = None

# Parameters (tunable)
MIN_SERMON_LENGTH_S = 10 * 60  # 10 minutes default
GAP_THRESHOLD_S = 2.0  # merge segments separated by <= 2s
KEYWORD_LIST = ["sermon", "scripture", "amen", "blessed", "gospel", "preach", "pray", "reading", "message"]
LENGTH_WEIGHT = 0.50
KEYWORD_WEIGHT = 0.30
VERIFY_WEIGHT = 0.20


def _load_wave_and_sr(state: Dict[str, Any]):
    if state.get("working_path") and os.path.exists(state["working_path"]):
        wav, sr = sf.read(state["working_path"], dtype="float32")
        return wav, sr
    if state.get("waveform") is not None and state.get("sr") is not None:
        return np.asarray(state["waveform"], dtype="float32"), int(state["sr"])
    raise FileNotFoundError("No working_path or waveform in state")


def _merge_segments(segments: List[Dict[str, Any]], gap_thresh: float = GAP_THRESHOLD_S) -> List[Dict[str, Any]]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    merged = []
    cur = {"start": segs[0]["start"], "end": segs[0]["end"], "text": segs[0].get("text", "")}
    for s in segs[1:]:
        if s["start"] - cur["end"] <= gap_thresh:
            cur["end"] = s["end"]
            cur["text"] += " " + s.get("text", "")
        else:
            merged.append(cur)
            cur = {"start": s["start"], "end": s["end"], "text": s.get("text", "")}
    merged.append(cur)
    return merged


def _keyword_density(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    count = 0
    for kw in KEYWORD_LIST:
        count += t.count(kw)
    # density = occurrences per 100 words (rough)
    words = max(1, len(t.split()))
    density = (count / words) * 100.0
    # normalize to 0..1 with a soft cap
    return min(1.0, density / 2.0)  # 2 occurrences per 100 words -> ~1.0


def _select_candidate_blocks(merged: List[Dict[str, Any]], wav_len_s: float, enrollment_registry: Optional[str],
                             wav: np.ndarray, sr: int, device: str, log_buffer: list) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Score each merged block and return a list of scored blocks and a debug summary.
    Each scored block: {start, end, text, length_s, keyword_density, verify_score, score}
    """
    scored = []
    debug = {"blocks_evaluated": 0}
    for m in merged:
        start = float(m.get("start", 0.0))
        end = float(m.get("end", start))
        length = max(0.0, end - start)
        if length <= 0:
            continue
        # length score normalized by wav_len_s (cap)
        length_score = min(1.0, length / max(1.0, wav_len_s))
        # keyword density
        kd = _keyword_density(m.get("text", ""))
        # speaker verification (optional)
        verify_score = 0.0
        if enrollment_registry and os.path.exists(enrollment_registry):
            try:
                s_sample = int(start * sr)
                e_sample = int(end * sr)
                seg_wav = wav[s_sample:e_sample]
                verify_score = _verify_speaker_with_registry(seg_wav, sr, enrollment_registry, device, log_buffer)
            except Exception as e:
                log_buffer.append(f"verify_block_error: {e}")
                verify_score = 0.0
        # composite score
        score = (LENGTH_WEIGHT * length_score) + (KEYWORD_WEIGHT * kd) + (VERIFY_WEIGHT * verify_score)
        score = max(0.0, min(1.0, score))
        scored.append({
            "start": start,
            "end": end,
            "text": m.get("text", ""),
            "length_s": length,
            "length_score": length_score,
            "keyword_density": kd,
            "verify_score": verify_score,
            "score": score
        })
        debug["blocks_evaluated"] += 1
    # sort by score descending
    scored_sorted = sorted(scored, key=lambda b: b["score"], reverse=True)
    return scored_sorted, debug


def _verify_speaker_with_registry(segment_waveform: np.ndarray, sr: int, enrollment_registry_path: str, device: str, log_buffer: list) -> float:
    """
    Placeholder for speaker verification. Returns a similarity score 0..1.
    If no registry or pyannote available, returns 0.0.
    """
    if not enrollment_registry_path or not os.path.exists(enrollment_registry_path):
        return 0.0
    try:
        with open(enrollment_registry_path, "r") as f:
            registry = json.load(f)
    except Exception:
        return 0.0
    # If pyannote available, compute embedding and compare to stored embeddings.
    if PyannoteInference is None:
        return 0.0
    try:
        # TODO: adapt to your pyannote usage and registry format
        inf = PyannoteInference("pyannote/embedding", device=device)
        emb = inf(segment_waveform, sample_rate=sr)
        # registry expected to contain normalized embeddings per preacher id
        best = 0.0
        for preacher_id, rec in registry.items():
            stored = np.asarray(rec.get("embedding", []), dtype="float32")
            if stored.size == 0:
                continue
            stored = stored / (np.linalg.norm(stored) + 1e-12)
            embn = emb / (np.linalg.norm(emb) + 1e-12)
            sim = float(np.dot(embn, stored))
            best = max(best, sim)
        return float(best)
    except Exception as e:
        log_buffer.append(f"speaker_verify_error: {e}")
        return 0.0


def remove_non_sermon_part(state: Dict[str, Any], log_buffer: list, enrollment_registry: str = None,
                           device: str = "cpu", test_mode: bool = True) -> Dict[str, Any]:
    """
    Main entry point for sermon extraction.
    - state: pipeline state dict (expects 'segments' list)
    - enrollment_registry: path to JSON with preacher embeddings (optional)
    - device: device string for embedding model
    - test_mode: if True, write outputs with '-sermon-test.wav' suffix to avoid overwriting canonical outputs
    """
    segments = state.get("segments") or []
    if not segments:
        log_buffer.append("remove_non_sermon_part: no segments available; skipping sermon extraction")
        state["sermon_selection"] = {"start_s": 0.0, "end_s": None, "score": 0.0, "reasons": ["no_segments"]}
        return state

    try:
        wav, sr = _load_wave_and_sr(state)
    except Exception as e:
        log_buffer.append(f"remove_non_sermon_part: load error: {e}")
        state["sermon_selection"] = {"start_s": 0.0, "end_s": None, "score": 0.0, "reasons": ["load_error"]}
        return state

    total_duration = len(wav) / float(sr) if sr else 0.0
    merged = _merge_segments(segments)

    # Score candidate blocks
    scored_blocks, debug = _select_candidate_blocks(merged, total_duration, enrollment_registry, wav, sr, device, log_buffer)
    if not scored_blocks:
        log_buffer.append("remove_non_sermon_part: no scored blocks; skipping")
        state["sermon_selection"] = {"start_s": 0.0, "end_s": None, "score": 0.0, "reasons": ["no_scored_blocks"]}
        return state

    # Choose best block (highest score) but ensure minimum length; if best is too short, try next best
    chosen = None
    for b in scored_blocks:
        if b["length_s"] >= MIN_SERMON_LENGTH_S:
            chosen = b
            break
    # If none meet min length, pick the longest block and mark as low_confidence
    if chosen is None:
        # pick block with max length
        longest = max(scored_blocks, key=lambda x: x["length_s"])
        chosen = longest
        chosen["low_confidence_reason"] = "below_min_length"

    start_s = float(chosen["start"])
    end_s = float(chosen["end"]) if chosen.get("end") is not None else total_duration
    final_score = float(chosen["score"])

    # Safety: ensure boundaries are within file
    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(wav), int(end_s * sr))
    if end_sample <= start_sample:
        log_buffer.append("remove_non_sermon_part: invalid sample boundaries; aborting trim")
        state["sermon_selection"] = {"start_s": 0.0, "end_s": None, "score": 0.0, "reasons": ["invalid_boundaries"]}
        return state

    # Write trimmed sermon file (use test suffix if test_mode)
    base = Path(state.get("working_path", "audio")).stem
    out_dir = Path(state.get("working_path", ".")).parent
    suffix = "-sermon-test.wav" if test_mode else "-sermon.wav"
    sermon_path = str(out_dir / f"{base}{suffix}")
    try:
        trimmed = wav[start_sample:end_sample]
        tmp = str(Path(sermon_path).with_suffix(sermon_path + ".tmp"))
        sf.write(tmp, trimmed, sr, subtype="PCM_16")
        os.replace(tmp, sermon_path)
        state["working_path"] = sermon_path
        state.setdefault("final_paths", {})["sermon"] = sermon_path
        reasons = ["scored_selection"]
        if chosen.get("verify_score", 0.0) > 0:
            reasons.append("speaker_verified")
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
        log_buffer.append(f"remove_non_sermon_part: wrote sermon file {sermon_path}")
    except Exception as e:
        log_buffer.append(f"remove_non_sermon_part: write failed: {e}")
        state["sermon_selection"] = {"start_s": start_s, "end_s": end_s, "score": final_score, "reasons": ["write_failed"]}

    # Attach debug info for audit
    state.setdefault("sermon_debug", {})["scored_blocks_count"] = len(scored_blocks)
    state.setdefault("sermon_debug", {})["top_blocks"] = scored_blocks[:3]
    state.setdefault("sermon_debug", {})["total_duration_s"] = total_duration

    return state
