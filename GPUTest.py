#!/usr/bin/env python3
"""
gpu_model_test.py

Usage examples:
  python gpu_model_test.py --test-whisper --whisper-model small --whisper-device 1
  python gpu_model_test.py --test-whisper --whisper-model medium --whisper-device 0 --test-embed --embed-device 1

Requirements:
  - Python 3.8+
  - nvidia-smi on PATH
  - Optional: faster-whisper, pyannote.audio, demucs installed in your env for full tests
"""

import argparse
import subprocess
import json
import time
import sys

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, encoding="utf8", stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.output

def list_gpus():
    out = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx, name, total, free = parts[0], parts[1], parts[2], parts[3]
            gpus.append({"index": int(idx), "name": name, "memory_total_mb": int(total), "memory_free_mb": int(free)})
    return gpus

def print_gpus(gpus):
    print("Detected GPUs:")
    for g in gpus:
        print(f"  GPU {g['index']}: {g['name']}  total={g['memory_total_mb']} MB  free={g['memory_free_mb']} MB")
    if not gpus:
        print("  No NVIDIA GPUs detected or nvidia-smi not available.")

def safe_sleep(sec=1.0):
    try:
        time.sleep(sec)
    except KeyboardInterrupt:
        pass

def test_whisper(model_name, device, compute_type="float16"):
    print(f"\nTesting Whisper model '{model_name}' on device cuda:{device} (compute_type={compute_type})")
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print("  faster_whisper not installed or failed to import:", e)
        return {"ok": False, "error": "faster_whisper_import_failed"}

    import torch
    before_free = get_free_gpu_mb(device)
    print(f"  free VRAM before load: {before_free} MB")

    try:
        t0 = time.time()
        model = WhisperModel(model_name, device=f"cuda:{device}", compute_type=compute_type)
        load_time = time.time() - t0
        after_free = get_free_gpu_mb(device)
        print(f"  loaded model in {load_time:.1f}s; free VRAM after load: {after_free} MB")
        # show torch memory stats if available
        try:
            print(f"  torch reserved: {torch.cuda.memory_reserved(device)} MB")
            print(f"  torch allocated: {torch.cuda.memory_allocated(device)} MB")
        except Exception:
            pass
        # cleanup
        del model
        torch.cuda.empty_cache()
        safe_sleep(0.5)
        freed = get_free_gpu_mb(device)
        print(f"  freed VRAM after unload: {freed} MB")
        return {"ok": True, "load_time_s": load_time, "free_after_load_mb": after_free}
    except RuntimeError as e:
        print("  RuntimeError during model load (likely OOM):", e)
        return {"ok": False, "error": "runtime_error", "detail": str(e)}
    except Exception as e:
        print("  Unexpected error during model load:", e)
        return {"ok": False, "error": "other", "detail": str(e)}

def test_pyannote(device):
    print(f"\nTesting pyannote audio embedding load on device cuda:{device}")
    try:
        import torch
        from pyannote.audio import Inference
    except Exception as e:
        print("  pyannote.audio not installed or failed to import:", e)
        return {"ok": False, "error": "pyannote_import_failed"}

    before = get_free_gpu_mb(device)
    print(f"  free VRAM before load: {before} MB")
    try:
        t0 = time.time()
        # This uses the default pretrained pipeline name; adjust if you have a specific model
        inf = Inference("pyannote/embedding", device=f"cuda:{device}")
        load_time = time.time() - t0
        after = get_free_gpu_mb(device)
        print(f"  loaded pyannote Inference in {load_time:.1f}s; free VRAM after load: {after} MB")
        del inf
        torch.cuda.empty_cache()
        safe_sleep(0.5)
        freed = get_free_gpu_mb(device)
        print(f"  freed VRAM after unload: {freed} MB")
        return {"ok": True, "load_time_s": load_time, "free_after_load_mb": after}
    except RuntimeError as e:
        print("  RuntimeError during pyannote load (likely OOM):", e)
        return {"ok": False, "error": "runtime_error", "detail": str(e)}
    except Exception as e:
        print("  Unexpected error during pyannote load:", e)
        return {"ok": False, "error": "other", "detail": str(e)}

def test_demucs(device):
    print(f"\nTesting Demucs model load on device cuda:{device}")
    try:
        import torch
        # demucs import may vary; try the common entrypoints
        from demucs import pretrained
    except Exception as e:
        print("  demucs not installed or failed to import:", e)
        return {"ok": False, "error": "demucs_import_failed"}

    before = get_free_gpu_mb(device)
    print(f"  free VRAM before load: {before} MB")
    try:
        t0 = time.time()
        # load a small demucs model
        model = pretrained.get_model("demucs_quantized") if hasattr(pretrained, "get_model") else pretrained.load_model("demucs")
        model.to(f"cuda:{device}")
        load_time = time.time() - t0
        after = get_free_gpu_mb(device)
        print(f"  loaded Demucs in {load_time:.1f}s; free VRAM after load: {after} MB")
        del model
        torch.cuda.empty_cache()
        safe_sleep(0.5)
        freed = get_free_gpu_mb(device)
        print(f"  freed VRAM after unload: {freed} MB")
        return {"ok": True, "load_time_s": load_time, "free_after_load_mb": after}
    except RuntimeError as e:
        print("  RuntimeError during Demucs load (likely OOM):", e)
        return {"ok": False, "error": "runtime_error", "detail": str(e)}
    except Exception as e:
        print("  Unexpected error during Demucs load:", e)
        return {"ok": False, "error": "other", "detail": str(e)}

def get_free_gpu_mb(gpu_index=0):
    out = run_cmd(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"])
    try:
        lines = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return lines[gpu_index]
    except Exception:
        return -1

def main():
    p = argparse.ArgumentParser(description="GPU model load tester")
    p.add_argument("--test-whisper", action="store_true")
    p.add_argument("--whisper-model", default="small", help="Whisper model name for faster-whisper")
    p.add_argument("--whisper-device", type=int, default=0)
    p.add_argument("--test-embed", action="store_true", help="Test pyannote embedding load")
    p.add_argument("--embed-device", type=int, default=0)
    p.add_argument("--test-demucs", action="store_true", help="Test Demucs load")
    p.add_argument("--demucs-device", type=int, default=0)
    args = p.parse_args()

    gpus = list_gpus()
    print_gpus(gpus)

    if args.test_whisper:
        res = test_whisper(args.whisper_model, args.whisper_device)
        print("Whisper test result:", json.dumps(res, indent=2))

    if args.test_embed:
        res = test_pyannote(args.embed_device)
        print("Pyannote test result:", json.dumps(res, indent=2))

    if args.test_demucs:
        res = test_demucs(args.demucs_device)
        print("Demucs test result:", json.dumps(res, indent=2))

    print("\nDone. If a load failed with OOM, try a smaller model or compute_type=float16/int8 or move that model to CPU.")

if __name__ == "__main__":
    main()
