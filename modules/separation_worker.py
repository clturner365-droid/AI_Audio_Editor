#!/usr/bin/env python3
"""
separation_worker.py

Persistent worker pinned to a single GPU for diarization, speaker verification,
vocal separation (Demucs/Spleeter), and the new pipeline steps:
- remove_singing
- remove_non_sermon_part

Designed to be launched with CUDA_VISIBLE_DEVICES set to the target GPU.
"""

import os
import time
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf

# Optional imports; adapt to your environment
try:
    from demucs import pretrained as demucs_pretrained
except Exception:
    demucs_pretrained = None

try:
    from pyannote.audio import Inference as PyannoteInference
except Exception:
    PyannoteInference = None

# GPU snapshot helper (same as transcription worker)
def get_gpu_memory_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"
        ], encoding="utf8")
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        info = []
        for line in lines:
            idx, name, total, free = [p.strip() for p in line.split(",")]
            info.append({"index": int(idx), "name": name, "memory_total_mb": int(total), "memory_free_mb": int(free)})
        return info
    except Exception:
        return []

def log_gpu_state(log_buffer, label):
    info = get_gpu_memory_info()
    ts = time.time()
    entry = {"time": ts, "label": label, "gpus": info}
    log_buffer.append(f"GPU_STATE: {json.dumps(entry)}")
    return entry

# Simple vocal separation wrapper (Demucs preferred)
def separate_vocals_demucs(wav_path: str, device: str, log_buffer: list):
    """
    Returns dict with keys: vocal_path, instrumental_path, vocal_reduction_estimate
    This is a minimal wrapper; adapt to your Demucs invocation or use a Python API.
    """
    base = Path(wav_path).stem
    out_dir = Path(wav_path).parent
    vocal_out = str(out_dir / f"{base}.vocal.wav")
    inst_out = str(out_dir / f"{base}.inst.wav")
    log_buffer.append(f"separate_vocals: demucs_call for {wav_path} device={device}")
    # If demucs_pretrained available, use it; otherwise call CLI if installed
    try:
        if demucs_pretrained is not None:
            model = demucs_pretrained.get_model("demucs_small")
            model.to(device)
            wav, sr = sf.read(wav_path, dtype="float32")
            # Demucs API usage varies; this is a placeholder for actual separation
            # Save placeholders for downstream steps
            sf.write(vocal_out, wav, sr)
            sf.write(inst_out, wav, sr)
        else:
            # fallback: call demucs CLI if available
            cmd = ["demucs", "-n", "demucs_small", "-o", str(out_dir), wav_path]
            subprocess.run(cmd, check=False)
        # crude vocal_reduction estimate placeholder
        vocal_reduction = 0.8
        return {"vocal_path": vocal_out, "inst_path": inst_out, "vocal_reduction": vocal_reduction}
    except Exception as e:
        log_buffer.append(f"separate_vocals_error: {e}")
        return {"vocal_path": None, "inst_path": None, "vocal_reduction": 0.0}

# Singing detection: energy-based on vocal stem
def detect_singing_intervals(vocal_path: str, sr: int = 16000, frame_ms: int = 200, energy_threshold: float = 0.01):
    try:
        wav, sr = sf.read(vocal_path, dtype="float32")
    except Exception:
        return []
    frame_len = int(sr * (frame_ms / 1000.0))
    intervals = []
    i = 0
    n = len(wav)
    singing = False
    start = 0.0
    while i < n:
        frame = wav[i:i+frame_len]
        energy = float(np.mean(frame * frame)) if len(frame) else 0.0
        t = i / sr
        if energy >= energy_threshold and not singing:
            singing = True
            start = t
        elif energy < energy_threshold and singing:
            end = t
            intervals.append((start, end))
            singing = False
        i += frame_len
    if singing:
        intervals.append((start, n / sr))
    return intervals

# Placeholder for remove_singing step
def remove_singing_step(state: dict, log_buffer: list, device: str, test_mode: bool):
    wav_path = state.get("working_path")
    if not wav_path or not os.path.exists(wav_path):
        log_buffer.append("remove_singing: no working_path found")
        return state
    log_buffer.append("remove_singing: start")
    log_gpu_state(log_buffer, "before_separation")
    sep = separate_vocals_demucs(wav_path, device, log_buffer)
    log_gpu_state(log_buffer, "after_separation")
    state.setdefault("singing_removal", {})
    state["singing_removal"].update({
        "vocal_path": sep.get("vocal_path"),
        "inst_path": sep.get("inst_path"),
        "vocal_reduction": sep.get("vocal_reduction")
    })
    # detect intervals
    if sep.get("vocal_path"):
        intervals = detect_singing_intervals(sep["vocal_path"])
        state["singing_removal"]["intervals"] = intervals
        state["singing_removal"]["spoken_overlap"] = 0.0  # placeholder; compute overlap with segments if available
    # apply removal: simple mute or replace with instrumental
    # For safety, write a new working copy with suffix
    base = Path(wav_path).stem
    out_dir = Path(wav_path).parent
    no_singing_path = str(out_dir / f"{base}-no-singing.wav")
    # naive copy for placeholder
    sf.write(no_singing_path, *sf.read(wav_path, dtype="float32"))
    state["working_path"] = no_singing_path
    state.setdefault("final_paths", {})["no_singing"] = no_singing_path
    log_buffer.append(f"remove_singing: wrote {no_singing_path}")
    return state

# Placeholder for remove_non_sermon_part step
def remove_non_sermon_part_step(state: dict, log_buffer: list, enrollment_registry: str, device: str, test_mode: bool):
    """
    Use diarization + speaker verification + transcript cues to select sermon region.
    This is a simplified placeholder that selects the longest speech region.
    """
    wav_path = state.get("working_path")
    if not wav_path or not os.path.exists(wav_path):
        log_buffer.append("remove_non_sermon_part: no working_path found")
        return state
    log_buffer.append("remove_non_sermon_part: start")
    # naive approach: if segments exist, pick longest contiguous speech block
    segments = state.get("segments") or []
    if not segments:
        log_buffer.append("remove_non_sermon_part: no segments; skipping")
        return state
    # find longest contiguous block by merging adjacent segments with small gaps
    merged = []
    gap_thresh = 2.0
    cur = segments[0].copy()
    for s in segments[1:]:
        if s["start"] - cur["end"] <= gap_thresh:
            cur["end"] = s["end"]
        else:
            merged.append(cur)
            cur = s.copy()
    merged.append(cur)
    # pick longest
    best = max(merged, key=lambda x: x["end"] - x["start"])
    start_s, end_s = best["start"], best["end"]
    # safety: enforce minimum sermon length
    min_len = 600  # 10 minutes
    if (end_s - start_s) < min_len:
        log_buffer.append("remove_non_sermon_part: selected region too short; skipping trim")
        state["sermon_selection"] = {"start_s": 0.0, "end_s": None, "score": 0.0}
        return state
    # write trimmed file
    wav, sr = sf.read(wav_path, dtype="float32")
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    trimmed = wav[start_sample:end_sample]
    base = Path(wav_path).stem
    out_dir = Path(wav_path).parent
    sermon_path = str(out_dir / f"{base}-sermon.wav")
    sf.write(sermon_path, trimmed, sr)
    state["working_path"] = sermon_path
    state.setdefault("final_paths", {})["sermon"] = sermon_path
    state["sermon_selection"] = {"start_s": start_s, "end_s": end_s, "score": 0.9, "reasons": ["longest_contiguous_speech"]}
    log_buffer.append(f"remove_non_sermon_part: wrote {sermon_path}")
    return state

def main_loop(watch_dir: str, out_dir: str, device: str, enrollment_registry: str, poll_s: int = 5):
    while True:
        try:
            for ready in Path(watch_dir).glob("*.done"):
                base = ready.stem
                wav_path = os.path.join(watch_dir, f"{base}.wav")
                if not os.path.exists(wav_path):
                    continue
                log_buffer = []
                state = {
                    "input_path": wav_path,
                    "working_path": wav_path,
                    "base": base,
                    "out_dir": out_dir,
                    "actions": []
                }
                # load segments if present
                seg_json = os.path.join(out_dir, f"{base}.segments.json")
                if os.path.exists(seg_json):
                    with open(seg_json, "r", encoding="utf8") as f:
                        segs = json.load(f).get("segments", [])
                        state["segments"] = segs
                # run singing removal
                state = remove_singing_step(state, log_buffer, device=device, test_mode=True)
                # run sermon extraction
                state = remove_non_sermon_part_step(state, log_buffer, enrollment_registry=enrollment_registry, device=device, test_mode=True)
                # write per-file actions log
                actions_path = os.path.join(out_dir, f"{base}.actions.json")
                _atomic_write_json(actions_path, {"actions": log_buffer})
                # mark done -> processed marker
                os.replace(str(ready), os.path.join(watch_dir, f"{base}.processed"))
        except Exception as e:
            print("worker_loop_error:", e)
        time.sleep(poll_s)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--enrollment-registry", default="/etc/sermon/enrollments.json")
    args = p.parse_args()
    main_loop(args.watch, args.out, args.device, args.enrollment_registry)
