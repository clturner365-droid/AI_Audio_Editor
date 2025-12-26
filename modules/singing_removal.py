#!/usr/bin/env python3
"""
modules/singing_removal.py

Provides `remove_singing(state, log_buffer, device, test_mode)`.

Behavior (safe defaults and placeholders):
- Accepts `state` with keys: 'working_path' (preferred) or 'waveform'+'sr'.
- Runs vocal separation (Demucs preferred) if available; falls back to Spleeter CLI or a no-op.
- Detects singing intervals on the vocal stem using short-time energy.
- Computes simple metrics: vocal_reduction (0..1) and spoken_overlap (0..1 placeholder).
- Produces a new working WAV with singing removed or replaced by instrumental stem.
- Updates state keys:
  - state["singing_removal"] = { intervals, vocal_path, inst_path, vocal_reduction, spoken_overlap, method }
  - state["working_path"] and state["final_paths"]["no_singing"] when a new file is written
- Never overwrites originals. Writes outputs next to working_path.
"""

import os
import tempfile
import subprocess
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

# Optional imports (Demucs / Spleeter). Use if available.
try:
    from demucs import pretrained as demucs_pretrained  # type: ignore
except Exception:
    demucs_pretrained = None

# Parameters
DEFAULT_ENERGY_THRESHOLD = 0.0005  # tuned for float32 audio; adjust as needed
MIN_SINGING_DURATION_S = 0.5
FRAME_MS = 200


def _write_wav_atomic(path: str, data: np.ndarray, sr: int) -> None:
    tmp = str(Path(path).with_suffix(path + ".tmp"))
    sf.write(tmp, data, sr, subtype="PCM_16")
    os.replace(tmp, path)


def _energy_intervals(wav: np.ndarray, sr: int, frame_ms: int = FRAME_MS, threshold: float = DEFAULT_ENERGY_THRESHOLD) -> List[Tuple[float, float]]:
    frame_len = int(sr * (frame_ms / 1000.0))
    if frame_len <= 0:
        return []
    n = len(wav)
    intervals: List[Tuple[float, float]] = []
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
    if state.get("working_path") and os.path.exists(state["working_path"]):
        wav, sr = sf.read(state["working_path"], dtype="float32")
        return wav, sr
    if state.get("waveform") is not None and state.get("sr") is not None:
        return np.asarray(state["waveform"], dtype="float32"), int(state["sr"])
    raise FileNotFoundError("No working_path or waveform in state")


def _run_demucs_separation(wav_path: str, out_dir: str, device: str, log_buffer: list) -> Dict[str, Any]:
    """
    Try to use demucs Python API if available; otherwise attempt CLI call.
    Returns dict with keys: vocal_path, inst_path, method.
    """
    base = Path(wav_path).stem
    vocal_out = str(Path(out_dir) / f"{base}.vocal.wav")
    inst_out = str(Path(out_dir) / f"{base}.inst.wav")
    try:
        if demucs_pretrained is not None:
            model = demucs_pretrained.get_model("demucs_small")
            model.to(device)
            wav, sr = sf.read(wav_path, dtype="float32")
            # NOTE: actual demucs API usage varies; this is a placeholder.
            # TODO: Replace with correct demucs inference call for your version.
            # For now, write copies so downstream steps can run.
            sf.write(vocal_out, wav, sr)
            sf.write(inst_out, wav, sr)
            log_buffer.append("demucs: used demucs_pretrained placeholder (no real separation).")
            return {"vocal_path": vocal_out, "inst_path": inst_out, "method": "demucs_placeholder"}
        # fallback to CLI if installed
        cmd = ["demucs", "-n", "demucs_small", "-o", out_dir, wav_path]
        subprocess.run(cmd, check=False)
        # CLI writes into out_dir/<model>/<wavname>/separated_stems
        # Attempt to find vocal and instrumental outputs (best-effort)
        # TODO: adapt to your demucs CLI output layout
        log_buffer.append("demucs: invoked CLI (best-effort).")
        return {"vocal_path": vocal_out, "inst_path": inst_out, "method": "demucs_cli"}
    except Exception as e:
        log_buffer.append(f"demucs_error: {e}")
        return {"vocal_path": None, "inst_path": None, "method": "demucs_failed"}


def _fallback_spleeter_cli(wav_path: str, out_dir: str, log_buffer: list) -> Dict[str, Any]:
    """
    Try spleeter 2stems CLI if available. Writes stems into out_dir/<basename>/.
    """
    base = Path(wav_path).stem
    vocal_out = str(Path(out_dir) / f"{base}.vocal.wav")
    inst_out = str(Path(out_dir) / f"{base}.inst.wav")
    try:
        cmd = ["spleeter", "separate", "-p", "spleeter:2stems", "-o", out_dir, wav_path]
        subprocess.run(cmd, check=False)
        log_buffer.append("spleeter: invoked CLI (best-effort).")
        return {"vocal_path": vocal_out, "inst_path": inst_out, "method": "spleeter_cli"}
    except Exception as e:
        log_buffer.append(f"spleeter_error: {e}")
        return {"vocal_path": None, "inst_path": None, "method": "spleeter_failed"}


def remove_singing(state: Dict[str, Any], log_buffer: list, *, device: str = "cpu", test_mode: bool = True) -> Dict[str, Any]:
    """
    Main entry point used by orchestrator.
    - state: pipeline state dict
    - log_buffer: list to append human-readable log lines
    - device: device string for model (e.g., 'cuda:1' or 'cpu')
    - test_mode: if True, write outputs with suffixes and avoid destructive actions
    """
    try:
        wav, sr = _load_wave_from_state(state)
    except Exception as e:
        log_buffer.append(f"remove_singing: load error: {e}")
        return state

    working_path = state.get("working_path") or ""
    out_dir = Path(working_path).parent if working_path else Path(".")
    base = Path(working_path).stem if working_path else "audio"

    # 1) run separation
    sep = _run_demucs_separation(working_path, str(out_dir), device, log_buffer)
    if sep.get("vocal_path") is None:
        # try spleeter fallback
        sep = _fallback_spleeter_cli(working_path, str(out_dir), log_buffer)

    state.setdefault("singing_removal", {})
    state["singing_removal"].update({
        "method": sep.get("method"),
        "vocal_path": sep.get("vocal_path"),
        "inst_path": sep.get("inst_path")
    })

    # 2) detect singing intervals on vocal stem
    intervals: List[Tuple[float, float]] = []
    vocal_path = sep.get("vocal_path")
    if vocal_path and os.path.exists(vocal_path):
        try:
            vocal_wav, vocal_sr = sf.read(vocal_path, dtype="float32")
            intervals = _energy_intervals(vocal_wav, vocal_sr)
            state["singing_removal"]["intervals"] = intervals
            log_buffer.append(f"remove_singing: detected {len(intervals)} singing intervals")
        except Exception as e:
            log_buffer.append(f"remove_singing: vocal read error: {e}")
            intervals = []
    else:
        log_buffer.append("remove_singing: no vocal stem available; skipping interval detection")
        state["singing_removal"]["intervals"] = []

    # 3) estimate vocal_reduction (placeholder: ratio of energy removed)
    try:
        total_energy = float(np.mean(wav * wav)) if wav.size else 0.0
        vocal_energy = 0.0
        if vocal_path and os.path.exists(vocal_path):
            v, _ = sf.read(vocal_path, dtype="float32")
            vocal_energy = float(np.mean(v * v)) if v.size else 0.0
        vocal_reduction = 0.0
        if total_energy > 0:
            vocal_reduction = min(1.0, vocal_energy / (total_energy + 1e-12))
            # convert to reduction estimate (higher is better)
            vocal_reduction = max(0.0, min(1.0, 1.0 - vocal_reduction))
        state["singing_removal"]["vocal_reduction"] = vocal_reduction
    except Exception as e:
        log_buffer.append(f"remove_singing: vocal_reduction error: {e}")
        state["singing_removal"]["vocal_reduction"] = 0.0

    # 4) compute spoken_overlap (placeholder: 0.0). If segments exist, compute overlap.
    spoken_overlap = 0.0
    segments = state.get("segments") or []
    if intervals and segments:
        # compute fraction of singing intervals that overlap with any speech segment
        overlap_count = 0
        total_singing_duration = 0.0
        for s0, s1 in intervals:
            total_singing_duration += (s1 - s0)
            for seg in segments:
                seg_s, seg_e = seg.get("start", 0.0), seg.get("end", 0.0)
                if not (s1 <= seg_s or s0 >= seg_e):
                    overlap_count += 1
                    break
        if total_singing_duration > 0:
            spoken_overlap = min(1.0, overlap_count / max(1.0, len(intervals)))
    state["singing_removal"]["spoken_overlap"] = spoken_overlap

    # 5) apply removal: prefer replacement with instrumental stem if available and spoken_overlap low
    new_wav = wav.copy()
    if sep.get("inst_path") and os.path.exists(sep.get("inst_path")) and intervals:
        try:
            inst_wav, inst_sr = sf.read(sep["inst_path"], dtype="float32")
            if inst_sr != sr:
                log_buffer.append("remove_singing: instrument stem sample rate mismatch; skipping replacement")
            else:
                for (s0, s1) in intervals:
                    s_sample = int(s0 * sr)
                    e_sample = int(s1 * sr)
                    # clamp
                    s_sample = max(0, s_sample)
                    e_sample = min(len(new_wav), e_sample)
                    if s_sample >= e_sample:
                        continue
                    # replace segment with instrumental stem slice (loop or pad as needed)
                    inst_slice = inst_wav[s_sample:e_sample]
                    if len(inst_slice) < (e_sample - s_sample):
                        # pad with zeros
                        pad = np.zeros((e_sample - s_sample - len(inst_slice),), dtype="float32")
                        inst_slice = np.concatenate([inst_slice, pad])
                    new_wav[s_sample:e_sample] = inst_slice[:(e_sample - s_sample)]
                log_buffer.append("remove_singing: replaced singing intervals with instrumental stem")
        except Exception as e:
            log_buffer.append(f"remove_singing: instrumental replacement failed: {e}")
            # fallback to mute
            for (s0, s1) in intervals:
                s_sample = int(s0 * sr)
                e_sample = int(s1 * sr)
                s_sample = max(0, s_sample)
                e_sample = min(len(new_wav), e_sample)
                if s_sample < e_sample:
                    new_wav[s_sample:e_sample] = 0.0
            log_buffer.append("remove_singing: muted singing intervals as fallback")
    else:
        # no instrumental stem: mute singing intervals
        for (s0, s1) in intervals:
            s_sample = int(s0 * sr)
            e_sample = int(s1 * sr)
            s_sample = max(0, s_sample)
            e_sample = min(len(new_wav), e_sample)
            if s_sample < e_sample:
                new_wav[s_sample:e_sample] = 0.0
        log_buffer.append("remove_singing: muted singing intervals (no instrumental stem)")

    # 6) write new working file
    out_path = str(Path(out_dir) / f"{base}-no-singing.wav")
    try:
        _write_wav_atomic(out_path, new_wav, sr)
        state["working_path"] = out_path
        state.setdefault("final_paths", {})["no_singing"] = out_path
        log_buffer.append(f"remove_singing: wrote no-singing file {out_path}")
    except Exception as e:
        log_buffer.append(f"remove_singing: write failed: {e}")

    return state
