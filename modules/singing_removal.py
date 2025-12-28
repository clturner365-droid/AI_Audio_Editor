#!/usr/bin/env python3
"""
modules/singing_removal.py

Mute-only singing removal.

Behavior:
- Loads audio from state["working_path"] or state["waveform"] + state["sr"].
- Detects singing intervals using short-time energy.
- Mutes those intervals directly in the waveform.
- Writes a new working WAV file with "-no-singing" suffix.
- Updates:
    state["singing_removal"] = {
        intervals,
        vocal_reduction,
        spoken_overlap,
        method="mute_only"
    }
    state["working_path"]
    state["final_paths"]["no_singing"]
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

DEFAULT_ENERGY_THRESHOLD = 0.0005
MIN_SINGING_DURATION_S = 0.5
FRAME_MS = 200


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _write_wav_atomic(path: str, data: np.ndarray, sr: int) -> None:
    """Safely write WAV by writing to a temp file then replacing."""
    tmp = str(Path(path).with_suffix(".tmp"))
    sf.write(tmp, data, sr, subtype="PCM_16")
    os.replace(tmp, path)


def _energy_intervals(
    wav: np.ndarray,
    sr: int,
    frame_ms: int = FRAME_MS,
    threshold: float = DEFAULT_ENERGY_THRESHOLD
) -> List[Tuple[float, float]]:
    """Detect high-energy intervals (proxy for singing)."""
    frame_len = int(sr * (frame_ms / 1000.0))
    if frame_len <= 0:
        return []

    n = len(wav)
    intervals = []
    singing = False
    start = 0.0
    i = 0

    while i < n:
        frame = wav[i:i + frame_len]
        energy = float(np.mean(frame * frame)) if frame.size else 0.0
        t = i / sr

        if energy >= threshold and not singing:
            singing = True
            start = t
        elif energy < threshold and singing:
            end = t
            if (end - start) >= MIN_SINGING_DURATION_S:
                intervals.append((start, end))
            singing = False

        i += frame_len

    if singing:
        end = n / sr
        if (end - start) >= MIN_SINGING_DURATION_S:
            intervals.append((start, end))

    return intervals


def _load_wave_from_state(state: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """Load audio from working_path or waveform+sr."""
    if state.get("working_path") and os.path.exists(state["working_path"]):
        wav, sr = sf.read(state["working_path"], dtype="float32")
        return wav, sr

    if state.get("waveform") is not None and state.get("sr") is not None:
        return np.asarray(state["waveform"], dtype="float32"), int(state["sr"])

    raise FileNotFoundError("No working_path or waveform in state")


# ---------------------------------------------------------
# Main mute-only singing removal
# ---------------------------------------------------------

def remove_singing(state: Dict[str, Any], log_buffer: list, *, device: str = "cpu") -> Dict[str, Any]:
    """Detect singing intervals and mute them."""
    try:
        wav, sr = _load_wave_from_state(state)
    except Exception as e:
        append_file_log(log_buffer, f"remove_singing: load error: {e}")
        return state

    working_path = state.get("working_path") or ""
    out_dir = Path(working_path).parent if working_path else Path(".")
    base = Path(working_path).stem if working_path else "audio"

    # 1) Detect singing intervals
    intervals = _energy_intervals(wav, sr)
    append_file_log(log_buffer, f"remove_singing: detected {len(intervals)} singing intervals")

    state.setdefault("singing_removal", {})
    state["singing_removal"]["intervals"] = intervals
    state["singing_removal"]["method"] = "mute_only"

    # 2) Estimate vocal_reduction (energy removed)
    try:
        total_energy = float(np.mean(wav * wav)) if wav.size else 0.0
        muted_energy = 0.0
        for (s0, s1) in intervals:
            s = int(s0 * sr)
            e = int(s1 * sr)
            s = max(0, s)
            e = min(len(wav), e)
            if s < e:
                seg = wav[s:e]
                muted_energy += float(np.mean(seg * seg))
        vocal_reduction = 0.0
        if total_energy > 0:
            vocal_reduction = min(1.0, muted_energy / (total_energy + 1e-12))
            vocal_reduction = max(0.0, min(1.0, vocal_reduction))
        state["singing_removal"]["vocal_reduction"] = vocal_reduction
    except Exception as e:
        append_file_log(log_buffer, f"remove_singing: vocal_reduction error: {e}")
        state["singing_removal"]["vocal_reduction"] = 0.0

    # 3) Compute spoken_overlap (optional metric)
    spoken_overlap = 0.0
    segments = state.get("segments") or []
    if intervals and segments:
        overlap_count = 0
        for (s0, s1) in intervals:
            for seg in segments:
                seg_s, seg_e = seg.get("start", 0.0), seg.get("end", 0.0)
                if not (s1 <= seg_s or s0 >= seg_e):
                    overlap_count += 1
                    break
        spoken_overlap = min(1.0, overlap_count / max(1.0, len(intervals)))
    state["singing_removal"]["spoken_overlap"] = spoken_overlap

    # 4) Mute singing intervals
    new_wav = wav.copy()
    for (s0, s1) in intervals:
        s = int(s0 * sr)
        e = int(s1 * sr)
        s = max(0, s)
        e = min(len(new_wav), e)
        if s < e:
            new_wav[s:e] = 0.0
    append_file_log(log_buffer, "remove_singing: muted singing intervals")

    # 5) Write new working file
    out_path = str(Path(out_dir) / f"{base}-no-singing.wav")
    try:
        _write_wav_atomic(out_path, new_wav, sr)
        state["working_path"] = out_path
        state.setdefault("final_paths", {})["no_singing"] = out_path
        append_file_log(log_buffer, f"remove_singing: wrote no-singing file {out_path}")
    except Exception as e:
        append_file_log(log_buffer, f"remove_singing: write failed: {e}")

    return state


# ---------------------------------------------------------
# Dispatcher wrapper
# ---------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher entry point for singing removal.
    """

    log_buffer = ctx["log_buffer"]
    device = ctx.get("gpu_for_singing_removal", "cpu")
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: remove_singing ===")

    try:
        state = remove_singing(state, log_buffer, device=device)
    except Exception as e:
        append_file_log(log_buffer, f"remove_singing failed: {e}")
        return state

    state.setdefault("actions", []).append({
        "step": "remove_singing",
        "time": time.time(),
        "intervals": len(state.get("singing_removal", {}).get("intervals", [])),
        "method": "mute_only"
    })

    if save_stepwise:
        maybe_save_step_audio("remove_singing", state, ctx)

    return state
