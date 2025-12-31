#!/usr/bin/env python3
"""
modules/vram_monitor.py

Per-WAV VRAM monitoring and degraded-mode management.

Features:
    - Monitor GPU0 and GPU1 available VRAM.
    - Log VRAM state per WAV.
    - Activate degraded mode for GPU1 when VRAM is low.
    - Persist degraded mode activation with a 10-day TTL.
    - Stop processing when:
        * GPU0 hits hard VRAM limit, or
        * GPU1 degraded mode TTL expires.
    - Expose dispatcher-ready run(state, ctx) entry point.

Files (project root, same as Progress.txt):
    - vram_state.json
    - degraded_mode.json
    - degraded_mode_expired.txt
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

VRAM_STATE_FILE = "vram_state.json"
DEGRADED_MODE_FILE = "degraded_mode.json"
DEGRADED_MODE_EXPIRED_FILE = "degraded_mode_expired.txt"

# 10 days in seconds
DEGRADED_MODE_TTL_SECONDS = 10 * 24 * 60 * 60


# ---------------------------------------------------------
# Helpers: time / JSON
# ---------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------
# Helpers: VRAM queries
# ---------------------------------------------------------

def _get_available_vram_mb(gpu_id: int) -> Optional[int]:
    """
    Returns available VRAM in MB for the given GPU, or None if unavailable.
    """
    if torch is None or not torch.cuda.is_available():
        return None

    try:
        torch.cuda.synchronize(device=gpu_id)
        stats = torch.cuda.mem_get_info(device=gpu_id)
        free_bytes = stats[0]
        return int(free_bytes / (1024 * 1024))
    except Exception:
        return None


# ---------------------------------------------------------
# Degraded mode state
# ---------------------------------------------------------

def _load_degraded_mode() -> Optional[Dict[str, Any]]:
    return _read_json(DEGRADED_MODE_FILE)


def _save_degraded_mode(reason: str) -> Dict[str, Any]:
    now = datetime.utcnow()
    expires = now + timedelta(seconds=DEGRADED_MODE_TTL_SECONDS)
    data = {
        "activated": now.isoformat(timespec="seconds") + "Z",
        "expires": expires.isoformat(timespec="seconds") + "Z",
        "reason": reason,
    }
    _write_json(DEGRADED_MODE_FILE, data)
    return data


def _degraded_mode_expired(dm: Dict[str, Any]) -> bool:
    try:
        expires_str = dm.get("expires")
        if not expires_str:
            return True
        expires = datetime.fromisoformat(expires_str.replace("Z", ""))
        return datetime.utcnow() >= expires
    except Exception:
        return True


def _write_degraded_mode_expired_note(dm: Dict[str, Any]) -> None:
    msg = (
        f"Degraded mode expired at {_now_iso()}\n"
        f"Original activation: {dm.get('activated')}\n"
        f"Reason: {dm.get('reason')}\n"
    )
    with open(DEGRADED_MODE_EXPIRED_FILE, "w", encoding="utf-8") as f:
        f.write(msg)


# ---------------------------------------------------------
# Main VRAM monitor
# ---------------------------------------------------------

def check_vram_and_update(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core VRAM logic.

    Returns a dict:
        {
            "continue": bool,          # False → stop batch
            "use_gpu0": bool | None,   # None = unchanged
            "use_gpu1": bool | None,   # None = unchanged
            "state": {...}             # vram_state.json contents
        }
    """

    logger = ctx.get("logger")
    file_index = ctx.get("file_index")

    # Defaults: do not change existing flags
    result = {
        "continue": True,
        "use_gpu0": None,
        "use_gpu1": None,
        "state": {},
    }

    # Query VRAM
    gpu0_free = _get_available_vram_mb(0)
    gpu1_free = _get_available_vram_mb(1)

    # Soft/hard thresholds (MB) – adjust if needed
    GPU0_SOFT = 2000
    GPU0_HARD = 1000

    GPU1_SOFT = 1500
    GPU1_HARD = 800

    vram_state = {
        "timestamp": _now_iso(),
        "file_index": file_index,
        "gpu0_available_mb": gpu0_free,
        "gpu1_available_mb": gpu1_free,
    }

    # Log basic VRAM state
    if logger:
        logger.info({
            "step": "vram_monitor",
            "event": "vram_state",
            "file_index": file_index,
            "gpu0_available_mb": gpu0_free,
            "gpu1_available_mb": gpu1_free,
        })

    # Handle GPU0 (no degraded mode, hard stop)
    if gpu0_free is not None:
        if gpu0_free <= GPU0_HARD:
            if logger:
                logger.error({
                    "step": "vram_monitor",
                    "event": "gpu0_hard_limit",
                    "gpu0_available_mb": gpu0_free,
                })
            vram_state["gpu0_status"] = "hard_limit"
            result["continue"] = False
        elif gpu0_free <= GPU0_SOFT:
            if logger:
                logger.warning({
                    "step": "vram_monitor",
                    "event": "gpu0_soft_warning",
                    "gpu0_available_mb": gpu0_free,
                })
            vram_state["gpu0_status"] = "soft_warning"
        else:
            vram_state["gpu0_status"] = "ok"

    # Handle GPU1 (degraded mode allowed)
    degraded_mode = _load_degraded_mode()

    if degraded_mode:
        # Check TTL
        if _degraded_mode_expired(degraded_mode):
            if logger:
                logger.error({
                    "step": "vram_monitor",
                    "event": "degraded_mode_expired",
                    "degraded_mode": degraded_mode,
                })
            _write_degraded_mode_expired_note(degraded_mode)
            vram_state["gpu1_status"] = "degraded_expired"
            result["continue"] = False
        else:
            # Degraded mode active, force CPU for GPU1 workloads
            vram_state["gpu1_status"] = "degraded_active"
            result["use_gpu1"] = False
            if logger:
                logger.info({
                    "step": "vram_monitor",
                    "event": "degraded_mode_active",
                    "degraded_mode": degraded_mode,
                })
    else:
        # No degraded mode yet; check thresholds
        if gpu1_free is not None:
            if gpu1_free <= GPU1_HARD:
                # Activate degraded mode
                dm = _save_degraded_mode(reason="gpu1_low_vram")
                vram_state["gpu1_status"] = "degraded_activated"
                result["use_gpu1"] = False

                if logger:
                    logger.warning({
                        "step": "vram_monitor",
                        "event": "degraded_mode_activated",
                        "gpu1_available_mb": gpu1_free,
                        "degraded_mode": dm,
                    })

                # Also write a simple timestamped note file
                note_name = f"degraded_mode_activated_{int(time.time())}.txt"
                with open(note_name, "w", encoding="utf-8") as f:
                    f.write(
                        f"Degraded mode activated at {_now_iso()}\n"
                        f"Reason: gpu1_low_vram\n"
                        f"GPU1 free: {gpu1_free} MB\n"
                    )

            elif gpu1_free <= GPU1_SOFT:
                vram_state["gpu1_status"] = "soft_warning"
                if logger:
                    logger.warning({
                        "step": "vram_monitor",
                        "event": "gpu1_soft_warning",
                        "gpu1_available_mb": gpu1_free,
                    })
            else:
                vram_state["gpu1_status"] = "ok"

    # Persist VRAM state
    _write_json(VRAM_STATE_FILE, vram_state)
    result["state"] = vram_state
    return result


# ---------------------------------------------------------
# Dispatcher wrapper
# ---------------------------------------------------------

def run(state: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatcher entry point.

    Called once per WAV, typically near the start of processing.
    Updates ctx flags for GPU usage and may signal that the batch
    should stop (via ctx["stop_batch"] = True).
    """

    logger = ctx.get("logger")

    result = check_vram_and_update(ctx)

    # Apply GPU usage flags to ctx if set
    if result["use_gpu0"] is not None:
        ctx["use_gpu0"] = result["use_gpu0"]
    if result["use_gpu1"] is not None:
        ctx["use_gpu1"] = result["use_gpu1"]

    # If continue is False, mark batch stop
    if not result["continue"]:
        ctx["stop_batch"] = True
        if logger:
            logger.error({
                "step": "vram_monitor",
                "event": "stop_batch_triggered",
                "reason": "vram_limits_or_degraded_ttl",
                "vram_state": result["state"],
            })

    # Attach last VRAM state to state for debugging if desired
    state.setdefault("system", {})["vram_state"] = result["state"]

    return state
