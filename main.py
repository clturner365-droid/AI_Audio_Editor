#!/usr/bin/env python3
"""
Main entry point for the sermon processing pipeline (rewritten version).

Responsibilities:
- Parse command-line arguments
- Initialize logging
- Perform GPU sanity checks (startup)
- Load configuration
- Initialize models (once per run)
- Load input list and progress
- Loop over files:
  - Process each file
  - Update progress
  - Run VRAM health monitor
  - Check control file for graceful stop
- Handle input list rollover
- Write summary information
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# These modules will be implemented as part of the rewrite.
from pipeline import (
  config,
  gpu_health,
  logging as logmod,
  progress,
  control,
  models,
  process_file,
)


def parse_args(argv=None):
  """
  Parse command-line arguments for the pipeline.
  """
  parser = argparse.ArgumentParser(
    description="Sermon processing pipeline (rewritten)."
  )

  parser.add_argument(
    "--input-list",
    required=True,
    type=Path,
    help="Path to text file containing list of input WAV files.",
  )

  parser.add_argument(
    "--output-dir",
    required=True,
    type=Path,
    help="Directory where all outputs and logs will be written.",
  )

  parser.add_argument(
    "--registry",
    required=True,
    type=Path,
    help="Path to speaker registry file.",
  )

  parser.add_argument(
    "--config",
    type=Path,
    default=None,
    help="Optional YAML configuration file.",
  )

  parser.add_argument(
    "--progress-file",
    type=Path,
    default=None,
    help="Optional path to progress tracking file (defaults to Progress.txt in output dir).",
  )

  parser.add_argument(
    "--control-file",
    type=Path,
    default=None,
    help="Optional path to control file (defaults to control.txt in output dir).",
  )

  parser.add_argument(
    "--summary-log",
    type=Path,
    default=None,
    help="Optional path to global summary log (defaults to Summary.log in output dir).",
  )

  parser.add_argument(
    "--only",
    nargs="*",
    default=None,
    help="Optional list of step names to run exclusively.",
  )

  parser.add_argument(
    "--skip",
    nargs="*",
    default=None,
    help="Optional list of step names to skip.",
  )

  parser.add_argument(
    "--save-stepwise",
    action="store_true",
    help="If set, save intermediate audio outputs for each major step.",
  )

  # GPU / device options
  parser.add_argument(
    "--gpu-whisper",
    type=str,
    default="cuda:0",
    help="Device for Whisper model (e.g., cuda:0 or cpu).",
  )

  parser.add_argument(
    "--gpu-vad",
    type=str,
    default="cuda:0",
    help="Device for VAD/segmentation model.",
  )

  parser.add_argument(
    "--gpu-embed",
    type=str,
    default="cuda:1",
    help="Device for speaker embedding model.",
  )

  parser.add_argument(
    "--gpu-demucs",
    type=str,
    default="cuda:1",
    help="Device for Demucs model.",
  )

  args = parser.parse_args(argv)
  return args


def resolve_paths(args):
  """
  Normalize and derive paths that depend on --output-dir defaults.
  """
  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)

  progress_file = args.progress_file or (output_dir / "Progress.txt")
  control_file = args.control_file or (output_dir / "control.txt")
  summary_log = args.summary_log or (output_dir / "Summary.log")

  return {
    "output_dir": output_dir,
    "progress_file": progress_file,
    "control_file": control_file,
    "summary_log": summary_log,
  }


def main(argv=None):
  # 1) Parse arguments
  args = parse_args(argv)
  paths = resolve_paths(args)

  output_dir = paths["output_dir"]
  progress_file = paths["progress_file"]
  control_file = paths["control_file"]
  summary_log = paths["summary_log"]

  # 2) Initialize logging
  logmod.init_logging(summary_log=summary_log)

  logmod.log_info(f"=== Pipeline run started at {datetime.now().isoformat()} ===")
  logmod.log_info(f"Using input list: {args.input_list}")
  logmod.log_info(f"Output directory: {output_dir}")
  logmod.log_info(f"Progress file: {progress_file}")
  logmod.log_info(f"Control file: {control_file}")
  logmod.log_info(f"Summary log: {summary_log}")

  # 3) Reset control file to 'continue' at startup
  control.write_initial_state(control_file)

  # 4) Load configuration
  cfg = config.load_config(args.config)

  # 5) GPU startup checks + baseline capture
  gpu_state = gpu_health.perform_startup_checks_and_capture_baseline(
    cfg=cfg,
    devices={
      "whisper": args.gpu_whisper,
      "vad": args.gpu_vad,
      "embed": args.gpu_embed,
      "demucs": args.gpu_demucs,
    },
    summary_logger=logmod,
  )

  # 6) Load all models once
  model_bundle = models.load_all_models(
    cfg=cfg,
    devices={
      "whisper": args.gpu_whisper,
      "vad": args.gpu_vad,
      "embed": args.gpu_embed,
      "demucs": args.gpu_demucs,
    },
    summary_logger=logmod,
  )

  # 7) Load input list
  input_files = config.load_input_list(args.input_list)
  if not input_files:
    logmod.log_error("Input list is empty. Nothing to do.")
    return 1

  # 8) Load progress
  start_index = progress.load_start_index(progress_file, len(input_files))
  logmod.log_info(f"Starting at index {start_index} of {len(input_files)}.")

  # 9) Main loop
  for index in range(start_index, len(input_files)):
    wav_path = Path(input_files[index])

    # Per-file logger
    per_file_logger = logmod.create_file_logger(output_dir=output_dir, wav_path=wav_path)
    per_file_logger.log_file_header(index=index, total=len(input_files))

    file_start_time = datetime.now()

    try:
      # a) Process the file
      result = process_file.process_single_file(
        wav_path=wav_path,
        output_dir=output_dir,
        cfg=cfg,
        models=model_bundle,
        gpu_state=gpu_state,
        per_file_logger=per_file_logger,
        global_logger=logmod,
        only_steps=args.only,
        skip_steps=args.skip,
        save_stepwise=args.save_stepwise,
      )

      # b) Update progress
      progress.update_progress(progress_file, index)

      # c) VRAM health monitor
      gpu_health.check_vram_health_after_file(
        gpu_state=gpu_state,
        cfg=cfg,
        per_file_logger=per_file_logger,
        global_logger=logmod,
      )

      # d) Compute elapsed time
      file_end_time = datetime.now()
      elapsed = (file_end_time - file_start_time).total_seconds()

      # e) Summary line
      logmod.write_summary_line(
        timestamp=file_end_time,
        wav_path=wav_path,
        result=result,
        gpu_state=gpu_state,
        elapsed_seconds=elapsed,
      )

      # f) Control file check
      if control.should_stop(control_file):
        logmod.log_warning("Graceful stop requested via control file. Halting after current file.")
        break

    except Exception as exc:
      per_file_logger.log_exception("Unhandled exception while processing file", exc)
      logmod.log_error(f"Unhandled exception on file {wav_path}: {exc}")
      return 1

    finally:
      per_file_logger.log_file_footer()

  # 10) Input list rollover
  if index == len(input_files) - 1:
    config.handle_input_list_rollover(
      input_list_path=args.input_list,
      progress_file=progress_file,
      global_logger=logmod,
    )

  logmod.log_info(f"=== Pipeline run finished at {datetime.now().isoformat()} ===")
  return 0


if __name__ == "__main__":
  sys.exit(main())
