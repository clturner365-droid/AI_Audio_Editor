#!/usr/bin/env python3
"""
modules/metadata_writer.py

Provides `prepare_and_write_metadata(state, config, log_buffer)`.

Behavior:
- Builds a canonical sidecar JSON with audit trail, actions, gpu_log, scores, decision placeholders.
- Writes sidecar atomically to config["logging"]["sidecar_dir"] or to the working directory.
- Optionally embeds a small safe subset of metadata into FLAC/MP3 files if:
    - config["auto_accept_mode"] is True
    - config["logging"]["embed_tags"] is True
    - decision in state indicates finalize (this module does not compute decision; it uses state)
- Always returns updated state with 'sidecar_path' set.
"""

import os
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

try:
    from mutagen.flac import FLAC
    from mutagen.mp3 import EasyMP3 as MP3
except Exception:
    FLAC = None
    MP3 = None


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(path) or "."
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    os.replace(tmp, path)


def _embed_tags_if_allowed(final_path: str, metadata: Dict[str, Any], config: Dict[str, Any], log_buffer: list) -> None:
    embed_allowed = config.get("auto_accept_mode", False) and config.get("logging", {}).get("embed_tags", False)
    decision_action = metadata.get("decision", {}).get("action", "")
    if not embed_allowed or decision_action != "finalize":
        log_buffer.append(f"embed_tags: skipped (embed_allowed={embed_allowed}, decision={decision_action})")
        return

    ext = Path(final_path).suffix.lower()
    tags = {
        "SERMON_START": str(metadata.get("sermon_selection", {}).get("start_s", "")),
        "SERMON_END": str(metadata.get("sermon_selection", {}).get("end_s", "")),
        "PIPELINE_VERSION": metadata.get("pipeline_version", ""),
        "DECISION": metadata.get("decision", {}).get("action", "")
    }
    try:
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

def prepare_and_write_metadata(state: Dict[str, Any], config: Dict[str, Any], log_buffer: list) -> Dict[str, Any]:
    """
    Build metadata object and write sidecar JSON atomically. Optionally embed tags.
    """
    ts = time.time()
    base = state.get("base") or Path(state.get("input_path", "audio")).stem
    sidecar_dir = config.get("logging", {}).get("sidecar_dir") or Path(state.get("out_dir", "."))
    sidecar_dir = Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = str(sidecar_dir / f"{base}.metadata.json")

    metadata = {
        "pipeline_version": config.get("pipeline_version", "unknown"),
        "input_path": state.get("input_path"),
        "working_path": state.get("working_path"),
        "final_paths": state.get("final_paths", {}),
        "original_metadata": state.get("original_metadata", {}),
        "speaker_corrections": state.get("speaker_corrections", []),
        "sermon_selection": state.get("sermon_selection", {}),
        "intro_outro": state.get("intro_outro", {}),
        "singing_removal": state.get("singing_removal", {}),
        "fingerprint": state.get("fp", {}),
        "scores": state.get("scores", {}),
        "actions": state.get("actions", []),
        "gpu_log": state.get("gpu_log", []),
        "decision": state.get("decision", {}),
        "qa": state.get("qa", {"required": False, "notes": []}),
        "timestamp": ts,
        "audit": state.get("audit", [])
    }

    try:
        _atomic_write_json(sidecar_path, metadata)
        log_buffer.append(f"prepare_and_write_metadata: sidecar written {sidecar_path}")
        state["sidecar_path"] = sidecar_path
    except Exception as e:
        log_buffer.append(f"prepare_and_write_metadata: sidecar write failed: {e}")

    # Optionally embed tags into final artifacts (only if allowed by config and decision)
    final_paths = state.get("final_paths", {}) or {}
    for name, path in final_paths.items():
        if path and os.path.exists(path):
            _embed_tags_if_allowed(path, metadata, config, log_buffer)

    return state

