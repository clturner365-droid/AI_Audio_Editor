#!/usr/bin/env python3
"""
modules/dispatcher.py

Central dispatcher for the sermon processing pipeline.

Order of operations per WAV file:

  1. ASR (Whisper)                           -> state["transcript"], confidences
  2. Sermon extraction                       -> state["sermon_selection"], trimmed audio
  3. Singing removal                         -> state["singing_removal"]
  4. Artifact checks                         -> state["artifact_checks"]
  5. Fingerprinting / identity               -> state["fingerprint"]
  6. Title generation (if missing)           -> state["sermon_title"]
  7. Intro/outro assembly + transcript shift -> state["working_path"], state["transcript"]
  8. Decision engine                         -> state["decision"], metadata sidecar
  9. Cleanup speaker queue                   -> fixes historical speaker names

All modules are expected to expose:

  def run(state, ctx) -> state
"""

import os
from typing import List, Dict, Any

from modules.logging_system import append_file_log

from modules import (
    asr_whisper,
    sermon_extractor,
    singing_removal,
    artifact_checks,
    fingerprint_engine,
    intro_outro,
    cleanup_speaker_queue,
    decision_engine,
    title_generator,
)

# ---------------------------------------------------------
# Global model caches
# ---------------------------------------------------------

_TITLE_MODEL = None


def _ensure_title_model(log_buffer):
    """
    Lazily load the CPU-only title model once for the entire process.
    """
    global _TITLE_MODEL
    if _TITLE_MODEL is None:
        append_file_log(log_buffer, "dispatcher: loading title generation model (CPU-only)...")
        _TITLE_MODEL = title_generator.load_title_model(log_buffer=log_buffer)
        append_file_log(log_buffer, "dispatcher: title generation model ready.")
    return _TITLE_MODEL


# ---------------------------------------------------------
# Per-file pipeline execution
# ---------------------------------------------------------

def _init_state_for_file(input_path: str, out_dir: str) -> Dict[str, Any]:
    base = os.path.splitext(os.path.basename(input_path))[0]
    state: Dict[str, Any] = {
        "input_path": input_path,
        "working_path": input_path,   # modules will update this as they go
        "base": base,
        "out_dir": out_dir,
        "actions": [],
        "scores": {},
    }
    return state


def _init_ctx_for_file(file_index: int, config: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "file_index": file_index,
        "log_buffer": [],
        "config": config,
    }
    return ctx


def _generate_title_if_missing(state: Dict[str, Any], ctx: Dict[str, Any]):
    """
    Dispatcher-level title generation step.

    If state["sermon_title"] is missing or empty, generate one from the transcript
    using the CPU-only title model and store it back into state.
    """
    log_buffer = ctx["log_buffer"]
    title = state.get("sermon_title")
    transcript = state.get("transcript")

    if title:
        append_file_log(log_buffer, f"dispatcher: sermon_title already present: {title}")
        return

    if not transcript:
        append_file_log(log_buffer, "dispatcher: no transcript available; cannot generate title.")
        return

    model = _ensure_title_model(log_buffer)
    generated = title_generator.generate_title_from_transcript(
        transcript=transcript.get("full_text", transcript) if isinstance(transcript, dict) else transcript,
        model=model,
        log_buffer=log_buffer,
    )
    state["sermon_title"] = generated
    append_file_log(log_buffer, f"dispatcher: generated sermon_title: {generated}")


def run_pipeline_for_file(input_path: str, file_index: int, config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """
    Run the full pipeline for a single WAV file.
    Returns the final state for that file.
    """
    state = _init_state_for_file(input_path, out_dir)
    ctx = _init_ctx_for_file(file_index, config)
    log_buffer = ctx["log_buffer"]

    append_file_log(log_buffer, f"=== Pipeline start for file_index={file_index}, input={input_path} ===")

    # 1. ASR (Whisper)
    state = asr_whisper.run(state, ctx)

    # 2. Sermon extraction
    state = sermon_extractor.run(state, ctx)

    # 3. Singing removal
    state = singing_removal.run(state, ctx)

    # 4. Artifact checks
    state = artifact_checks.run(state, ctx)

    # 5. Fingerprinting / identity
    state = fingerprint_engine.run(state, ctx)

    # 6. Title generation (if missing)
    _generate_title_if_missing(state, ctx)

    # 7. Intro/outro + transcript shift
    state = intro_outro.run(state, ctx)

    # 8. Decision engine (sidecar + decision)
    state = decision_engine.run(state, ctx)

    # 9. Cleanup speaker queue (out-of-band historical fixups)
    state = cleanup_speaker_queue.run(state, ctx)

    append_file_log(log_buffer, f"=== Pipeline end for file_index={file_index} ===")

    # Attach logs to state so caller can persist them if desired
    state["log_buffer"] = log_buffer
    return state


# ---------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------

def run_pipeline(input_paths: List[str], config: Dict[str, Any], out_dir: str) -> List[Dict[str, Any]]:
    """
    Run the pipeline for a list of input WAV paths.
    Returns a list of final state dicts, one per file.
    """
    results: List[Dict[str, Any]] = []

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for idx, input_path in enumerate(input_paths):
        state = run_pipeline_for_file(input_path=input_path, file_index=idx, config=config, out_dir=out_dir)
        results.append(state)

    return results
