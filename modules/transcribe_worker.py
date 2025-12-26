#!/usr/bin/env python3
"""
transcribe_worker.py

Persistent transcription worker pinned to a single GPU. Loads Whisper once,
processes files from a watch directory, writes segments and transcript, logs GPU state,
and writes intermediate artifacts. Designed to be launched with CUDA_VISIBLE_DEVICES set.
"""

import os
import time
import json
import argparse
import subprocess
from pathlib import Path

# Replace with your actual model loader (faster-whisper or whisper.cpp wrapper)
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

from prepare_and_write_metadata import _atomic_write_json  # reuse atomic writer if needed

# GPU snapshot helper
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

def load_whisper_model(model_name: str, device: str, compute_type: str, log_buffer: list):
    log_buffer.append(f"loading_whisper: model={model_name} device={device} compute_type={compute_type}")
    if WhisperModel is None:
        raise RuntimeError("WhisperModel not available in environment")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    log_buffer.append("whisper_loaded")
    return model

def transcribe_file(model, wav_path: str, out_dir: str, log_buffer: list):
    base = Path(wav_path).stem
    seg_out = os.path.join(out_dir, f"{base}.segments.json")
    txt_out = os.path.join(out_dir, f"{base}.txt")
    # Example using faster-whisper API
    try:
        segments, info = model.transcribe(wav_path, beam_size=5, vad_filter=True)
        # write segments and transcript
        segs = []
        transcript_lines = []
        for seg in segments:
            segs.append({"start": seg.start, "end": seg.end, "text": seg.text})
            transcript_lines.append(seg.text)
        _atomic_write_json(seg_out, {"segments": segs, "info": info})
        with open(txt_out, "w", encoding="utf8") as f:
            f.write("\n".join(transcript_lines))
        log_buffer.append(f"transcription_written: {seg_out}, {txt_out}")
        return {"segments": segs, "transcript": " ".join(transcript_lines), "info": info}
    except Exception as e:
        log_buffer.append(f"transcription_error: {e}")
        raise

def main_loop(watch_dir: str, out_dir: str, model_name: str, device: str, compute_type: str, poll_s: int = 5):
    log_buffer = []
    model = None
    try:
        model = load_whisper_model(model_name, device=device, compute_type=compute_type, log_buffer=log_buffer)
    except Exception as e:
        log_buffer.append(f"model_load_failed: {e}")
        raise

    while True:
        try:
            for ready in Path(watch_dir).glob("*.ready.json"):
                base = ready.stem.replace(".ready", "")
                wav_path = os.path.join(watch_dir, f"{base}.wav")
                if not os.path.exists(wav_path):
                    continue
                # per-file log
                file_log = []
                file_log.append(f"start_processing: {wav_path}")
                log_gpu_state(file_log, "before_transcribe")
                # transcribe
                result = transcribe_file(model, wav_path, out_dir, file_log)
                log_gpu_state(file_log, "after_transcribe")
                # write per-file action log
                actions_path = os.path.join(out_dir, f"{base}.actions.json")
                _atomic_write_json(actions_path, {"actions": file_log})
                # rename ready -> processing -> done markers as needed (caller policy)
                os.replace(str(ready), os.path.join(watch_dir, f"{base}.processing"))
                os.replace(os.path.join(watch_dir, f"{base}.processing"), os.path.join(watch_dir, f"{base}.done"))
        except Exception as e:
            log_buffer.append(f"worker_loop_error: {e}")
        time.sleep(poll_s)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", required=True, help="Directory to watch for .ready.json and .wav")
    p.add_argument("--out", required=True, help="Output directory for transcripts and segments")
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--compute_type", default="float16")
    args = p.parse_args()
    main_loop(args.watch, args.out, args.model, args.device, args.compute_type)
