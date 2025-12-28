"""
process_file.py

This module defines the per-file pipeline dispatcher.
It runs a sequence of named steps, each with a hierarchical step ID.

Responsibilities:
- Define ordered list of steps
- Provide a clean dispatcher loop
- Support --only and --skip filtering
- Provide per-step logging and timing
- Pass metadata between steps
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Step:
  id: str          # hierarchical ID, e.g. "audio.load"
  func: callable   # function implementing the step
  description: str # human-readable description


def process_single_file(
  wav_path,
  output_dir,
  cfg,
  models,
  gpu_state,
  per_file_logger,
  global_logger,
  only_steps=None,
  skip_steps=None,
  save_stepwise=False,
):
  """
  Run all pipeline steps for a single WAV file.
  Returns a result object used by the summary log.
  """

  # Metadata object passed between steps
  state = {
    "wav_path": wav_path,
    "output_dir": output_dir,
    "cfg": cfg,
    "models": models,
    "gpu_state": gpu_state,
    "save_stepwise": save_stepwise,
    "results": {},   # step outputs
  }

  # Define the ordered list of steps (placeholder functions for now)
  STEPS = [
    Step(id="audio.load",        func=step_audio_load,        description="Load WAV audio"),
    Step(id="audio.normalize",   func=step_audio_normalize,   description="Normalize audio"),
    Step(id="vad.detect",        func=step_vad_detect,        description="Voice activity detection"),
    Step(id="vad.merge",         func=step_vad_merge,         description="Merge VAD segments"),
    Step(id="whisper.transcribe",func=step_whisper_transcribe,description="Transcribe speech with Whisper"),
    Step(id="speaker.embed",     func=step_speaker_embed,     description="Compute speaker embedding"),
    Step(id="speaker.match",     func=step_speaker_match,     description="Match speaker to registry"),
    Step(id="demucs.separate",   func=step_demucs_separate,   description="Source separation with Demucs"),
    Step(id="metadata.write",    func=step_metadata_write,    description="Write metadata and JSON"),
    Step(id="output.finalize",   func=step_output_finalize,   description="Finalize output WAV"),
  ]

  # Normalize filters
  only_set = set(only_steps) if only_steps else None
  skip_set = set(skip_steps) if skip_steps else set()

  # Dispatcher loop
  for step in STEPS:

    # Skip filtering
    if only_set and step.id not in only_set:
      per_file_logger.log(f"[{step.id}] Skipped (not in --only)")
      continue

    if step.id in skip_set:
      per_file_logger.log(f"[{step.id}] Skipped (in --skip)")
      continue

    # Run step
    per_file_logger.log(f"[{step.id}] Starting: {step.description}")
    start_time = datetime.now()

    try:
      step.func(state, per_file_logger)
    except Exception as exc:
      per_file_logger.log_exception(f"[{step.id}] ERROR", exc)
      raise

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    per_file_logger.log(f"[{step.id}] Completed in {elapsed:.2f}s")

  # Return final result for summary log
  return state["results"]


# ---------------------------------------------------------------------------
# Placeholder step functions (we will fill these in later)
# ---------------------------------------------------------------------------

def step_audio_load(state, log):
  log.log("  (audio.load placeholder)")
  state["results"]["audio_loaded"] = True


def step_audio_normalize(state, log):
  log.log("  (audio.normalize placeholder)")


def step_vad_detect(state, log):
  log.log("  (vad.detect placeholder)")


def step_vad_merge(state, log):
  log.log("  (vad.merge placeholder)")


def step_whisper_transcribe(state, log):
  log.log("  (whisper.transcribe placeholder)")
  state["results"]["transcript"] = "placeholder transcript"


def step_speaker_embed(state, log):
  log.log("  (speaker.embed placeholder)")


def step_speaker_match(state, log):
  log.log("  (speaker.match placeholder)")
  state["results"]["speaker"] = "UNKNOWN"
  state["results"]["confidence"] = 0.0


def step_demucs_separate(state, log):
  log.log("  (demucs.separate placeholder)")


def step_metadata_write(state, log):
  log.log("  (metadata.write placeholder)")


def step_output_finalize(state, log):
  log.log("  (output.finalize placeholder)")
