#!/usr/bin/env python3
"""
main.py
End-to-end pipeline orchestrator for sermon processing with a minimal
"--save-stepwise" toggle that writes numbered WAVs after each executed step.

This version adds:
 - remove_singing
 - remove_non_sermon_part
 - prepare_and_write_metadata (sidecar + optional tag embed)
 - per-file GPU snapshots appended to state['gpu_log']
 - conservative behavior: originals are never overwritten; working copies are used
"""

import os
import json
import time
import argparse
from pathlib import Path
import subprocess

import soundfile as sf

# Module imports (assumes modules/ is on PYTHONPATH)
from modules.logging_system import append_file_log
from modules.audio_loader import load_audio
from modules.intro_outro_removal import remove_intros_outros
from modules.noise_reduction import reduce_noise
from modules.silence_trim import trim_silence
from modules.loudness_normalization import normalize_loudness   # used earlier in pipeline
from modules.transcript import generate_transcript
from modules.fingerprint_utils import (
    extract_fingerprint,
    compare_fingerprints,
    update_fingerprint_if_safe
)
from modules.save_outputs import save_output_files
from modules.singing_removal import remove_singing
from modules.sermon_extraction import remove_non_sermon_part
from modules.metadata_writer import prepare_and_write_metadata
# Outro pipeline modules (14A–14D)
from modules.tts_outro_generator import generate_dynamic_outro       
from modules.tts_audio_generator import generate_tts_audio           
from modules.loudness_normalizer import normalize_loudness as normalize_outro_loudness  
from modules.final_audio_assembler import assemble_final_audio       

# Pipeline step names (must match the function names used below)
ALL_STEPS = [
    "load_audio",
    "remove_intros_outros",
    "reduce_noise",
    "trim_silence",
    "normalize_loudness",
    "generate_transcript",
    "remove_singing",
    "remove_non_sermon_part",
    "extract_fingerprint",
    "compare_fingerprints",
    "update_fingerprint_if_safe",
    "generate_outro_text",        
    "generate_outro_audio",       
    "normalize_outro_audio",      
    "assemble_final_audio",       
    "prepare_and_write_metadata",   
    "save_output_files",
]

# Default processing parameters
TARGET_LUFS = -16.0
DEFAULT_SAMPLE_RATE = 16000


def parse_args():
    p = argparse.ArgumentParser(description="Sermon DSP pipeline")
    p.add_argument("--input", "-i", required=True, help="Input audio file path")
    p.add_argument("--output_dir", "-o", required=True, help="Directory to write outputs")
    p.add_argument("--registry", "-r", default="registry.json", help="Path to registry JSON")
    p.add_argument("--config", "-c", default="production_config.yaml", help="Path to production config YAML")
    p.add_argument("--only", nargs="*", default=None, help="Run only these steps (names from ALL_STEPS)")
    p.add_argument("--skip", nargs="*", default=[], help="Steps to skip")
    p.add_argument("--save-stepwise", action="store_true", help="Save audio after each executed step as numbered files")
    p.add_argument("--gpu-for-transcribe", default="cuda:0", help="Device string for transcription (Whisper)")
    p.add_argument("--gpu-for-separation", default="cuda:1", help="Device string for separation/diarization")
    p.add_argument("--enrollment-registry", default="/etc/sermon/enrollments.json", help="Speaker enrollment registry")
    return p.parse_args()


def should_run(step, only_list, skip_list):
    if only_list is not None:
        return step in only_list
    return step not in skip_list


def _save_stepwise_file(waveform, sr, out_dir, base_name, step_index):
    """
    Writes: {out_dir}/{base_name}-{step_index}.wav (16-bit PCM)
    Returns the path written.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"{base_name}-{step_index}.wav")
    sf.write(path, waveform, sr, subtype="PCM_16")
    return path


def _load_registry(path, log_buffer):
    if not path:
        append_file_log(log_buffer, "No registry path provided; starting with empty registry.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        append_file_log(log_buffer, f"Loaded registry from {path}.")
        return registry
    except FileNotFoundError:
        append_file_log(log_buffer, f"Registry file not found at {path}; creating new registry.")
        return {}
    except Exception as e:
        append_file_log(log_buffer, f"Failed to load registry: {e}")
        return {}


def _save_registry(path, registry, log_buffer):
    if not path:
        append_file_log(log_buffer, "No registry path provided; skipping registry save.")
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        append_file_log(log_buffer, f"Registry saved to {path}.")
    except Exception as e:
        append_file_log(log_buffer, f"Failed to save registry: {e}")


def _make_basename(input_path):
    p = Path(input_path)
    return p.stem


def get_gpu_memory_info():
    """
    Returns list of dicts: [{'index':0,'name':'...','memory_total_mb':..., 'memory_free_mb':...}, ...]
    If nvidia-smi is not available, returns [].
    """
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


def process_file(input_path, output_dir, registry_path, config, only_list=None, skip_list=None, save_stepwise=False,
                 gpu_for_transcribe="cuda:0", gpu_for_separation="cuda:1", enrollment_registry=None):
    """
    Runs the full pipeline for a single input file with stepwise-save support.
    All writes are to working copies; originals are never overwritten.
    """
    log_buffer = []
    start_time = time.time()
    append_file_log(log_buffer, f"Processing started for: {input_path}")

    registry = _load_registry(registry_path, log_buffer)
    canonical_names = registry.get("canonical_names", list(registry.get("fingerprints", {}).keys()))

    state = {
        "waveform": None,
        "sr": DEFAULT_SAMPLE_RATE,
        "transcript": "",
        "fp": None,
        "actions": [],
        "gpu_log": [],
        "original_metadata": {}
    }
    base_name = _make_basename(input_path)
    step_index = 0

    def maybe_save(step_name):
        nonlocal step_index
        if save_stepwise and state.get("waveform") is not None:
            step_index += 1
            p = _save_stepwise_file(state["waveform"], state["sr"], output_dir, base_name, step_index)
            append_file_log(log_buffer, f"Saved step {step_index} ({step_name}) -> {p}")
            state.setdefault("actions", []).append({"step": step_name, "time": time.time(), "saved_path": p})

    def log_gpu_snapshot(label):
        info = get_gpu_memory_info()
        entry = {"time": time.time(), "label": label, "gpus": info}
        state.setdefault("gpu_log", []).append(entry)
        append_file_log(log_buffer, f"GPU_SNAPSHOT {label}: {info}")

    # Make a working copy path (do not modify original)
    working_dir = Path(output_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    working_copy_path = str(working_dir / f"{base_name}-working.wav")
    # Copy original to working copy (read-only originals preserved)
    try:
        if not os.path.exists(working_copy_path):
            # read and write to ensure consistent sample rate if needed
            wf, sr = load_audio(input_path, log_buffer)
            state["original_metadata"] = original_md
            sf.write(working_copy_path, wf, sr, subtype="PCM_16")
            append_file_log(log_buffer, f"Created working copy at {working_copy_path}")
    except Exception as e:
        append_file_log(log_buffer, f"Failed to create working copy: {e}")
        raise

    # 1) load_audio
    step = "load_audio"
    if should_run(step, only_list, skip_list):
        waveform, sr = load_audio(working_copy_path, log_buffer)
        state["waveform"], state["sr"] = waveform, sr
        append_file_log(log_buffer, f"Loaded audio length: {len(waveform)} samples @ {sr} Hz")
        state.setdefault("actions", []).append({"step": step, "time": time.time()})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 2) remove_intros_outros
    step = "remove_intros_outros"
    if should_run(step, only_list, skip_list):
        state["waveform"] = remove_intros_outros(state["waveform"], state["sr"], log_buffer)
        append_file_log(log_buffer, f"After intro/outro removal: {len(state['waveform'])} samples")
        state.setdefault("actions", []).append({"step": step, "time": time.time()})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 3) reduce_noise
    step = "reduce_noise"
    if should_run(step, only_list, skip_list):
        state["waveform"] = reduce_noise(state["waveform"], state["sr"], log_buffer)
        append_file_log(log_buffer, f"After noise reduction: {len(state['waveform'])} samples")
        state.setdefault("actions", []).append({"step": step, "time": time.time()})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 4) trim_silence
    step = "trim_silence"
    if should_run(step, only_list, skip_list):
        state["waveform"] = trim_silence(state["waveform"], state["sr"], log_buffer)
        append_file_log(log_buffer, f"After silence trim: {len(state['waveform'])} samples")
        state.setdefault("actions", []).append({"step": step, "time": time.time()})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 5) normalize_loudness
    step = "normalize_loudness"
    if should_run(step, only_list, skip_list):
        state["waveform"] = normalize_loudness(state["waveform"], state["sr"], TARGET_LUFS, log_buffer)
        append_file_log(log_buffer, f"After loudness normalization: {len(state['waveform'])} samples")
        state.setdefault("actions", []).append({"step": step, "time": time.time()})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 6) generate_transcript
    step = "generate_transcript"
    if should_run(step, only_list, skip_list):
        # log GPU snapshot before transcription model load
        log_gpu_snapshot("before_transcribe")
        state["transcript"], state["segments"], state["transcript_confidences"] = generate_transcript(
            state["waveform"], state["sr"], log_buffer, device=gpu_for_transcribe
        )
        log_gpu_snapshot("after_transcribe")
        append_file_log(log_buffer, f"Transcript length: {len(state['transcript'])} characters")
        state.setdefault("actions", []).append({"step": step, "time": time.time()})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 7) remove_singing (NEW)
    step = "remove_singing"
    if should_run(step, only_list, skip_list):
        log_gpu_snapshot("before_separation")
        state = remove_singing(state, log_buffer, device=gpu_for_separation, test_mode=save_stepwise)
        log_gpu_snapshot("after_separation")
        append_file_log(log_buffer, f"After singing removal: waveform length {len(state.get('waveform', []))}")
        state.setdefault("actions", []).append({"step": step, "time": time.time(), "details": state.get("singing_removal", {})})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 8) remove_non_sermon_part (NEW)
    step = "remove_non_sermon_part"
    if should_run(step, only_list, skip_list):
        log_gpu_snapshot("before_sermon_extraction")
        state = remove_non_sermon_part(state, log_buffer, enrollment_registry, device=gpu_for_separation, test_mode=save_stepwise)
        log_gpu_snapshot("after_sermon_extraction")
        append_file_log(log_buffer, f"After sermon extraction: waveform length {len(state.get('waveform', []))}")
        state.setdefault("actions", []).append({"step": step, "time": time.time(), "details": state.get("sermon_selection", {})})
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 9) extract_fingerprint (v2)
    step = "extract_fingerprint"
    if should_run(step, only_list, skip_list):
        from modules.fingerprint_engine_v2 import extract_fingerprint_v2

        state["fp"] = extract_fingerprint_v2(state["waveform"], state["sr"], log_buffer)
        append_file_log(log_buffer, f"v2: Step 9 summary: fp={'none' if state['fp'] is None else 'vector'}")

        state.setdefault("actions", []).append({
            "step": step,
            "time": time.time()
        })
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")


    # 10) determine_speaker_identity (v2)
    step = "compare_fingerprints"
    decided_name = None
    best_match = None
    confidence = 0.0

    if should_run(step, only_list, skip_list):
        from modules.speaker_identity_v2 import determine_identity_and_match

        original_meta_name = None
        try:
            original_meta_name = state.get("original_metadata", {}).get("IART")
        except Exception:
            original_meta_name = None

        input_path = state.get("input_path", "")

        decided_name, best_match, confidence = determine_identity_and_match(
            registry=registry,
            canonical_names=canonical_names,
            fp_new=state["fp"],
            original_meta_name=original_meta_name,
            input_path=input_path,
            log_buffer=log_buffer,
        )

        append_file_log(
            log_buffer,
            f"v2: Step 10 summary: decided={decided_name}, "
            f"best_match={best_match}, conf={confidence:.3f}"
        )

        state["decided_speaker"] = decided_name
        state["fp_confidence"] = confidence

        state.setdefault("actions", []).append({
            "step": step,
            "time": time.time(),
            "result": {
                "decided_name": decided_name,
                "best_match": best_match,
                "confidence": confidence
            }
        })
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")


    # 11) update_fingerprint_if_safe (v2)
    step = "update_fingerprint_if_safe"
    update_action = None

    if should_run(step, only_list, skip_list):
        from modules.speaker_identity_v2 import update_fingerprint_for_decided_speaker

        if save_stepwise:
            append_file_log(
                log_buffer,
                "Save-stepwise test mode: skipping registry update to avoid changing fingerprints (v2)."
            )
            update_action = "test_mode_skip"
        else:
            decided_name = state.get("decided_speaker")
            multi = False  # or your multi-speaker detection flag

            registry, update_action = update_fingerprint_for_decided_speaker(
                registry=registry,
                decided_name=decided_name,
                fp_new=state["fp"],
                confidence=state.get("fp_confidence", 0.0),
                multi=multi,
                log_buffer=log_buffer,
             )

        append_file_log(
            log_buffer,
            f"v2: Step 11 summary: update_action={update_action}, "
            f"speaker={state.get('decided_speaker')}"
        )

        state.setdefault("actions", []).append({
            "step": step,
            "time": time.time(),
            "result": update_action
        })
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 12) generate_outro_text (NEW)
    step = "generate_outro_text"
    if should_run(step, only_list, skip_list):
      try:
          # Generate the dynamic outro text based on metadata + duration rules
          from modules.tts_outro_generator import generate_dynamic_outro

          # Metadata should already be in state from step 13
          metadata = state.get("metadata", {})

          outro_text = generate_dynamic_outro(metadata)
          state["outro_text"] = outro_text  # store for TTS step later

          append_file_log(log_buffer, f"Generated outro_text: {outro_text}")
          state.setdefault("actions", []).append({"step": step, "time": time.time()})

      except Exception as e:
          append_file_log(log_buffer, f"Error in {step}: {e}")
          state.setdefault("errors", []).append({"step": step, "error": str(e)})
      else:
          append_file_log(log_buffer, f"Skipped {step}")

    # 13) generate_outro_audio (NEW - Coqui TTS)
    step = "generate_outro_audio"
    if should_run(step, only_list, skip_list):
        try:
            from modules.tts_audio_generator import generate_tts_audio

            outro_text = state.get("outro_text")
            if not outro_text:
                raise ValueError("No outro_text found in state. Step 14A must run first.")

            outro_audio = generate_tts_audio(outro_text, config=config, log_buffer=log_buffer)
            state["outro_audio"] = outro_audio

            append_file_log(log_buffer, "generate_outro_audio completed")
            state.setdefault("actions", []).append({"step": step, "time": time.time()})

        except Exception as e:
            append_file_log(log_buffer, f"Error in {step}: {e}")
            state.setdefault("errors", []).append({"step": step, "error": str(e)})
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 14) normalize_outro_audio (NEW - calls loudness_normalizer)
    step = "normalize_outro_audio"
    if should_run(step, only_list, skip_list):
        try:
            from modules.loudness_normalizer import normalize_loudness

            outro_audio = state.get("outro_audio")
            sermon_audio = state.get("sermon_audio")

            if outro_audio is None:
                raise ValueError("No outro_audio found in state. Step 14B must run before 14C.")

            if sermon_audio is None:
                raise ValueError("No sermon_audio found in state. Sermon must be processed before 14C.")

            append_file_log(log_buffer, "Normalizing outro audio loudness to match sermon audio")

            normalized_outro = normalize_loudness(
                target_audio=outro_audio,
                reference_audio=sermon_audio,
                log_buffer=log_buffer
            )

            state["outro_audio_normalized"] = normalized_outro

            append_file_log(log_buffer, "normalize_outro_audio completed")
            state.setdefault("actions", []).append({"step": step, "time": time.time()})

        except Exception as e:
            append_file_log(log_buffer, f"Error in {step}: {e}")
            state.setdefault("errors", []).append({"step": step, "error": str(e)})
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 15) assemble_final_audio (NEW - calls external module)
    step = "assemble_final_audio"
    if should_run(step, only_list, skip_list):
        try:
            from modules.final_audio_assembler import assemble_final_audio

            intro_audio = state.get("intro_audio")  # optional
            sermon_audio = state.get("sermon_audio")
            outro_audio = state.get("outro_audio_normalized")

            if sermon_audio is None:
                raise ValueError("assemble_final_audio: sermon_audio missing")

            if outro_audio is None:
                raise ValueError("assemble_final_audio: outro_audio_normalized missing")

            append_file_log(log_buffer, "Calling final_audio_assembler to build final audio")

            # Call the external module to assemble the final audio
            final_audio = assemble_final_audio(
                intro_audio=intro_audio,
                sermon_audio=sermon_audio,
                outro_audio=outro_audio,
                log_buffer=log_buffer
            )

            # Store for Step 14 (save_output_files)
            state["final_audio"] = final_audio

            append_file_log(log_buffer, "assemble_final_audio completed")
            state.setdefault("actions", []).append({"step": step, "time": time.time()})

        except Exception as e:
            append_file_log(log_buffer, f"Error in {step}: {e}")
            state.setdefault("errors", []).append({"step": step, "error": str(e)})
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # 16) prepare_and_write_metadata (NEW)
        step = "prepare_and_write_metadata"
        if should_run(step, only_list, skip_list):
            # prepare metadata and write sidecar JSON; embedding tags only if auto_accept_mode and decision finalize
           state["input_path"] = input_path
           state["working_path"] = str(working_copy_path)
           state["final_paths"] = state.get("final_paths", {})
           state = prepare_and_write_metadata(state, config, log_buffer)
           append_file_log(log_buffer, "prepare_and_write_metadata completed")
           state.setdefault("actions", []).append({"step": step, "time": time.time()})
        else:
           append_file_log(log_buffer, f"Skipped {step}")

    # 17) save_output_files
    step = "save_output_files"
    saved = {}
    if should_run(step, only_list, skip_list):
        saved = save_output_files(
            output_dir,
            base_name,
            state["waveform"],
            state["sr"],
            state["transcript"],
            {
                "input_path": input_path,
                "best_match": best_match,
                "confidence": confidence,
                "update_action": update_action,
                "updated_name": updated_name,
                "processing_time_s": round(time.time() - start_time, 2)
            },
            log_buffer
        )

        append_file_log(log_buffer, f"Saved outputs: {saved}")

        # NEW: capture final WAV path
        final_wav_path = saved.get("wav")
        state["final_wav_path"] = final_wav_path

        # NEW: write metadata into the final WAV
        prepare_and_write_metadata(state, config, log_buffer, final_wav_path)

        state.setdefault("actions", []).append({
            "step": step,
            "time": time.time(),
            "saved": saved
        })
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # ---------------------------------------------------------
    # 18) cleanup_speaker_queue
    # This step checks the speaker_resolution_queue.txt file.
    # If the last line contains a RESOLVED entry, it will:
    #   - find all UNKNOWN entries for that numeric speaker
    #   - update embedded WAV metadata for those files
    #   - update their sidecar JSON files
    #   - remove all processed lines from the queue
    #   - leave only unresolved entries in the queue
    # If the last line is not RESOLVED, nothing happens.
    # ---------------------------------------------------------
    step = "cleanup_speaker_queue"
    if should_run(step, only_list, skip_list):
        try:
            from modules.cleanup_speaker_queue import run_cleanup
            run_cleanup(log_buffer)
            append_file_log(log_buffer, f"Completed {step}")
        except Exception as e:
            append_file_log(log_buffer, f"{step}_error: {e}")

        state.setdefault("actions", []).append({
            "step": step,
            "time": time.time()
        })
        maybe_save(step)
    else:
        append_file_log(log_buffer, f"Skipped {step}")

    # Save registry back to disk if not in test mode
    if not save_stepwise:
        _save_registry(registry_path, registry, log_buffer)
    else:
        append_file_log(log_buffer, "Save-stepwise active: registry save skipped.")

    append_file_log(log_buffer, "Processing complete.")
                 
    # Write pipeline log
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(output_dir, f"{base_name}.pipeline.log")
        with open(log_path, "a", encoding="utf-8") as lf:
            for line in log_buffer:
                lf.write(line + "\n")
        append_file_log(log_buffer, f"Log written to {log_path}")
    except Exception as e:
        append_file_log(log_buffer, f"Failed to write log file: {e}")

    # return state for caller inspection
    return {"saved": saved, "registry": registry, "log": log_buffer, "state": state}


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load config (yaml) if present
    config = {}
    try:
        import yaml
        with open(args.config, "r") as cf:
            config = yaml.safe_load(cf)
    except Exception:
        config = {}

    result = process_file(
        input_path=args.input,
        output_dir=args.output_dir,
        registry_path=args.registry,
        config=config,
        only_list=args.only,
        skip_list=args.skip,
        save_stepwise=args.save_stepwise,
        gpu_for_transcribe=args.gpu_for_transcribe,
        gpu_for_separation=args.gpu_for_separation,
        enrollment_registry=args.enrollment_registry
    )
    # Short summary to stdout
    print("Processing finished.")
    print("Saved files:", result["saved"])
    print("Registry keys:", list(result["registry"].keys()))


if __name__ == "__main__":
    main()










