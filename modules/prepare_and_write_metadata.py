#!/usr/bin/env python3
"""
prepare_and_write_metadata.py

Writes an atomic sidecar JSON for each processed file and optionally embeds
a small, safe subset of metadata into FLAC/MP3 when allowed. Always writes
the sidecar; embedding is conditional on `embed_tags` and file format support.

Usage:
  from prepare_and_write_metadata import prepare_and_write_metadata, embed_tags_into_file
  state = prepare_and_write_metadata(state, config, log_buffer, final_wav_path)
"""

import json
import os
import tempfile
import time
from typing import Dict, Any

try:
    from mutagen.flac import FLAC
    from mutagen.mp3 import EasyMP3 as MP3
except Exception:
    FLAC = None
    MP3 = None

def merge_metadata(original_md, corrected_md):
    """
    Merge original embedded metadata with corrected metadata.

    Rules:
    - Start with original metadata (BSI tags, RIFF INFO, etc.)
    - Override only fields present in corrected metadata
    - Never delete or null out original fields
    - Return a clean, final metadata dictionary
    """
    if original_md is None:
        original_md = {}
    if corrected_md is None:
        corrected_md = {}

    final = dict(original_md)

    for key, value in corrected_md.items():
        if value not in (None, "", []):
            final[key] = value

    return final

# Atomic write helper
def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    os.replace(tmp, path)

def _safe_timestamp() -> float:
    return time.time()

def embed_tags_into_file(final_path: str, metadata: Dict[str, Any], log_buffer: list) -> None:
    """
    Embed a small subset of metadata into FLAC or MP3 files.
    Sidecar JSON remains the authoritative record.
    """
    try:
        ext = os.path.splitext(final_path)[1].lower()
        tags = {
            "SERMON_START": str(metadata.get("sermon_selection", {}).get("start_s", "")),
            "SERMON_END": str(metadata.get("sermon_selection", {}).get("end_s", "")),
            "PIPELINE_VERSION": metadata.get("pipeline_version", ""),
            "DECISION": metadata.get("decision", {}).get("action", "")
        }
        if ext == ".flac" and FLAC is not None:
            audio = FLAC(final_path)
            for k, v in tags.items():
                if v != "":
                    audio[k] = v
            audio.save()
            log_buffer.append(f"embed_tags: wrote FLAC tags to {final_path}")
        elif ext in (".mp3", ".mpeg") and MP3 is not None:
            audio = MP3(final_path)
            for k, v in tags.items():
                if v != "":
                    audio[k] = v
            audio.save()
            log_buffer.append(f"embed_tags: wrote MP3 tags to {final_path}")
        else:
            log_buffer.append(f"embed_tags: unsupported format or mutagen missing for {final_path}")
    except Exception as e:
        log_buffer.append(f"embed_tags_error: {e}")

def prepare_and_write_metadata(state: Dict[str, Any], config: Dict[str, Any], log_buffer: list) -> Dict[str, Any]:
    """
    Build final metadata object, write sidecar JSON atomically, and optionally embed tags.
    - state: pipeline state with keys like input_path, working_path, final_paths, actions, scores, gpu_log, decision
    - config: production config dict (see production_config.yaml)
    - log_buffer: list to append human-readable log lines
    Returns updated state (metadata written path added).
    """
    ts = _safe_timestamp()
    base = state.get("base") or os.path.splitext(os.path.basename(state.get("input_path", "unknown")))[0]
    out_dir = state.get("out_dir") or config.get("logging", {}).get("sidecar_dir", ".")
    sidecar_path = os.path.join(out_dir, f"{base}.metadata.json")

    metadata = {
        "pipeline_version": config.get("pipeline_version", "unknown"),
        "input_path": state.get("input_path"),
        "working_path": state.get("working_path"),
        "final_paths": state.get("final_paths", {}),
        # We will merge original + corrected later, so don't embed original here
        # (we remove this field from the corrected metadata block)
        "speaker_corrections": state.get("speaker_corrections", []),
        "sermon_selection": state.get("sermon_selection", {}),
        "intro_outro": state.get("intro_outro", {}),
        "singing_removal": state.get("singing_removal", {}),
        "fingerprint": state.get("fingerprint", {}),
        "scores": state.get("scores", {}),
        "actions": state.get("actions", []),
        "gpu_log": state.get("gpu_log", []),
        "decision": state.get("decision", {}),
        "qa": state.get("qa", {"required": False, "notes": []}),
        "timestamp": ts,
        "audit": state.get("audit", [])
    }

# --- MERGE ORIGINAL + CORRECTED METADATA ---
original_md = state.get("original_metadata", {})
corrected_md = metadata

final_md = merge_metadata(original_md, corrected_md)

# Replace metadata with merged version
metadata = final_md

# Store merged metadata in state for downstream use
state["metadata"] = final_md

# Always write sidecar atomically
    try:
        _atomic_write_json(sidecar_path, final_md)
        log_buffer.append(f"sidecar_written: {sidecar_path}")
        state.setdefault("sidecar_path", sidecar_path)
    except Exception as e:
        log_buffer.append(f"sidecar_write_error: {e}")

    # Embed tags only if allowed by config and decision
    embed_allowed = config.get("auto_accept_mode", False) and config.get("logging", {}).get("embed_tags", False)
    decision_action = state.get("decision", {}).get("action", "")
    if embed_allowed and decision_action == "finalize":
        final_paths = state.get("final_paths", {})
        # embed into each final artifact if format supported
        for name, path in final_paths.items():
            try:
                embed_tags_into_file(path, final_md, log_buffer)
            except Exception as e:
                log_buffer.append(f"embed_tags_exception: {e}")

    return state


