#!/usr/bin/env python3
"""
main.py

Entry point for the sermon processing pipeline.

Design:
    - Flat dispatcher: main() calls each module's run(state, ctx).
    - Per-WAV loop with shared ctx and per-file state.
    - VRAM monitoring via modules.vram_monitor.
    - No resume/progress.txt logic wired (by your choice).
    - Logs per WAV and overall progress.

Expected module interface:
    def run(state: dict, ctx: dict) -> dict
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

# ---------------------------------------------------------
# Module imports
# ---------------------------------------------------------

from modules import (
    vram_monitor,
    audio_loader,
    demucs,
    separation_worker,
    singing_removal,
    noise_reduction,
    trim_silence,
    loudness_normalization,
    transcribe_worker,
    transcript,
    sermon_extraction,
    intro_outro_removal,
    contextual_rules,
    speaker_identity,
    metadata,
    registry,
    cleanup_speaker_queue,
    title_generator,
    tts_audio_generator,
    intro_outro,
    final_wav_writer,
    scripture_book_normalizer,
    chapter_verse_normalizer,
    html_linker,
    text_html_export,
)

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------
# State / ctx helpers
# ---------------------------------------------------------

def init_state_for_file(input_path: str, out_dir: str) -> Dict[str, Any]:
    base = os.path.splitext(os.path.basename(input_path))[0]
    return {
        "input_path": input_path,
        "base": base,
        "out_dir": out_dir,
        "actions": [],
        "scores": {},
    }


def init_ctx_for_file(
    file_index: int,
    logger: logging.Logger,
    debug: bool,
    save_stepwise: bool,
) -> Dict[str, Any]:
    return {
        "file_index": file_index,
        "logger": logger,
        "debug": debug,
        "save_stepwise": save_stepwise,
        "use_gpu0": True,
        "use_gpu1": True,
        "stop_batch": False,
    }


# ---------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------

def run_pipeline_for_file(
    wav_path: str,
    file_index: int,
    out_dir: str,
    logger: logging.Logger,
    debug: bool,
    save_stepwise: bool,
) -> Dict[str, Any]:

    state = init_state_for_file(wav_path, out_dir)
    ctx = init_ctx_for_file(file_index, logger, debug, save_stepwise)

    logger.info({
        "step": "main",
        "event": "file_start",
        "file_index": file_index,
        "input_path": wav_path,
    })

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # 0. VRAM monitor (per-WAV control)
    state = vram_monitor.run(state, ctx)
    if ctx.get("stop_batch"):
        logger.error({
            "step": "main",
            "event": "stop_batch_from_vram",
            "file_index": file_index,
        })
        return state

    # STAGE 1 — Load & pre-clean audio
    state = audio_loader.run(state, ctx)
    state = demucs.run(state, ctx)
    state = separation_worker.run(state, ctx)
    state = singing_removal.run(state, ctx)
    state = noise_reduction.run(state, ctx)
    state = trim_silence.run(state, ctx)
    state = loudness_normalization.run(state, ctx)

    # STAGE 2 — First transcript pass
    state = transcribe_worker.run(state, ctx)
    state = transcript.run(state, ctx)

    # STAGE 3 — Transcript-driven audio steps
    state = sermon_extraction.run(state, ctx)
    state = intro_outro_removal.run(state, ctx)
    state = contextual_rules.run(state, ctx)

    # STAGE 4 — Speaker identity, metadata, title
    state = speaker_identity.run(state, ctx)
    state = metadata.run(state, ctx)
    state = registry.run(state, ctx)
    state = cleanup_speaker_queue.run(state, ctx)
    state = title_generator.run(state, ctx)

    # STAGE 5 — Intro/outro TTS + final audio assembly
    state = tts_audio_generator.run(state, ctx)
    state = intro_outro.run(state, ctx)
    state = final_wav_writer.run(state, ctx)

    # STAGE 6 — Final transcript + scripture + HTML
    state = transcribe_worker.run(state, ctx)   # optional second pass; module should handle reuse if desired
    state = transcript.run(state, ctx)
    state = scripture_book_normalizer.run(state, ctx)
    state = chapter_verse_normalizer.run(state, ctx)
    state = html_linker.run(state, ctx)
    state = text_html_export.run(state, ctx)

    logger.info({
        "step": "main",
        "event": "file_complete",
        "file_index": file_index,
        "input_path": wav_path,
    })

    return state


# ---------------------------------------------------------
# Batch runner
# ---------------------------------------------------------

def run_batch(
    input_paths: List[str],
    out_dir: str,
    log_level: str = "INFO",
    debug: bool = False,
    save_stepwise: bool = False,
) -> List[Dict[str, Any]]:

    logger = setup_logger(log_level)
    logger.info({
        "step": "main",
        "event": "batch_start",
        "num_files": len(input_paths),
        "out_dir": out_dir,
    })

    results: List[Dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)

    for idx, wav_path in enumerate(input_paths):
        if not os.path.exists(wav_path):
            logger.error({
                "step": "main",
                "event": "missing_input",
                "file_index": idx,
                "input_path": wav_path,
            })
            continue

        state = run_pipeline_for_file(
            wav_path=wav_path,
            file_index=idx,
            out_dir=out_dir,
            logger=logger,
            debug=debug,
            save_stepwise=save_stepwise,
        )
        results.append(state)

        if state.get("system", {}).get("vram_state") and state.get("system", {}).get("vram_state", {}).get("gpu0_status") == "hard_limit":
            logger.error("Stopping batch due to GPU0 hard VRAM limit.")
            break

        if state.get("system", {}).get("vram_state") and state.get("system", {}).get("vram_state", {}).get("gpu1_status") in ("degraded_expired",):
            logger.error("Stopping batch due to degraded mode TTL expiration.")
            break

    logger.info({
        "step": "main",
        "event": "batch_complete",
        "processed_files": len(results),
    })

    return results


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sermon processing pipeline")
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Input WAV file(s)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in modules",
    )
    parser.add_argument(
        "--save-stepwise",
        action="store_true",
        help="Enable stepwise audio saving where supported",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    input_paths = args.input
    out_dir = args.out_dir
    log_level = args.log_level
    debug = bool(args.debug)
    save_stepwise = bool(args.save_stepwise)

    run_batch(
        input_paths=input_paths,
        out_dir=out_dir,
        log_level=log_level,
        debug=debug,
        save_stepwise=save_stepwise,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
