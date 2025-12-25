# decision_engine.py
"""
Decision engine for automated finalize/quarantine logic.

Usage:
  from decision_engine import evaluate_and_decide
  state = {...}  # pipeline state after processing steps
  config = load_config("/etc/sermon_pipeline/production_config.yaml")
  updated_state = evaluate_and_decide(state, config)
"""

import json
import os
import time
import math
import tempfile
import subprocess

# -------------------------
# Helpers
# -------------------------
def now_ts():
    return time.time()

def write_sidecar_atomic(path, data):
    dirpath = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    os.replace(tmp, path)

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

def safe_div(a, b):
    try:
        return a / b if b else 0.0
    except Exception:
        return 0.0

# -------------------------
# Per-check metric functions
# -------------------------
def metric_asr_confidence(state):
    """
    Expect state['transcript_confidences'] as list of per-segment mean confidences (0..1)
    or state['asr_mean_confidence'] as a single float.
    """
    if "asr_mean_confidence" in state:
        return float(state["asr_mean_confidence"])
    confs = state.get("transcript_confidences") or []
    if confs:
        return float(sum(confs) / len(confs))
    return 0.0

def metric_sermon_confidence(state):
    """
    Expect state['sermon_selection'] with keys 'score' (0..1) if available.
    Otherwise compute a fallback from speaker dominance, keyword density, and speech/music ratio.
    """
    sel = state.get("sermon_selection") or {}
    if "score" in sel:
        return float(sel["score"])
    # fallback heuristics
    speaker_dom = float(state.get("speaker_dominance", 0.0))  # 0..1
    keyword_density = float(state.get("keyword_density", 0.0))  # 0..1
    speech_music = float(state.get("speech_music_ratio", 1.0))  # 0..1
    # weighted fallback
    return (0.5 * speaker_dom) + (0.3 * keyword_density) + (0.2 * speech_music)

def metric_singing_quality(state):
    """
    Expect state['singing_removal'] with 'vocal_reduction' (0..1) and 'spoken_overlap' (0..1).
    Higher vocal_reduction and lower spoken_overlap => better score.
    """
    s = state.get("singing_removal") or {}
    vocal_reduction = float(s.get("vocal_reduction", 0.0))
    spoken_overlap = float(s.get("spoken_overlap", 0.0))
    # score = vocal_reduction * (1 - spoken_overlap)
    return vocal_reduction * (1.0 - spoken_overlap)

def metric_artifact_score(state):
    """
    Return 0..1 where 1 means no artifacts. Expect state['artifact_checks'] with keys:
    'clipping_ratio' (0..1), 'silence_spike' (0..1), 'rms_drop_db' (positive number).
    """
    a = state.get("artifact_checks") or {}
    clipping = float(a.get("clipping_ratio", 0.0))
    silence = float(a.get("silence_spike", 0.0))
    rms_drop_db = float(a.get("rms_drop_db", 0.0))
    # penalize clipping and silence; convert rms_drop_db to 0..1 penalty (cap at 30 dB)
    rms_penalty = min(rms_drop_db / 30.0, 1.0)
    penalty = max(clipping, silence, rms_penalty)
    return max(0.0, 1.0 - penalty)

def metric_fingerprint_penalty(state):
    """
    If fingerprint indicates duplicate or unsafe update, return penalty in 0..1.
    Expect state['fingerprint'] with 'duplicate_score' (0..1) and 'allow_update' boolean.
    """
    f = state.get("fingerprint") or {}
    dup = float(f.get("duplicate_score", 0.0))
    allow = bool(f.get("allow_update", True))
    if allow:
        return 0.0
    # penalty proportional to duplicate score
    return dup

# -------------------------
# Composite score
# -------------------------
def compute_composite_score(state, config):
    # weights from config
    w_asr = config["weights"].get("asr_confidence", 0.30)
    w_sermon = config["weights"].get("sermon_confidence", 0.30)
    w_singing = config["weights"].get("singing_quality", 0.20)
    w_artifact = config["weights"].get("artifact_score", 0.10)
    w_fp = config["weights"].get("fingerprint_penalty", 0.10)

    asr = metric_asr_confidence(state)
    sermon = metric_sermon_confidence(state)
    singing = metric_singing_quality(state)
    artifact = metric_artifact_score(state)
    fp_pen = metric_fingerprint_penalty(state)

    # normalize and combine; fingerprint is a penalty so subtract
    composite = (
        (w_asr * asr) +
        (w_sermon * sermon) +
        (w_singing * singing) +
        (w_artifact * artifact)
    ) - (w_fp * fp_pen)

    # clamp 0..1
    composite = max(0.0, min(1.0, composite))
    # attach metrics to state for audit
    state.setdefault("scores", {})
    state["scores"].update({
        "asr_confidence": asr,
        "sermon_confidence": sermon,
        "singing_quality": singing,
        "artifact_score": artifact,
        "fingerprint_penalty": fp_pen,
        "composite_score": composite
    })
    return composite

# -------------------------
# Decision logic
# -------------------------
def finalize_decision(state, config, log_buffer=None):
    """
    Apply thresholds and decide finalize vs quarantine vs accept-with-flags.
    Updates state['decision'] and writes sidecar JSON.
    """
    if log_buffer is None:
        log_buffer = []

    # compute composite if not present
    composite = state.get("scores", {}).get("composite_score")
    if composite is None:
        composite = compute_composite_score(state, config)

    ts = now_ts()
    decision = {
        "time": ts,
        "composite_score": composite,
        "action": None,
        "reasons": []
    }

    # thresholds
    auto_accept = config["thresholds"]["AUTO_ACCEPT_THRESHOLD"]
    quarantine = config["thresholds"]["QUARANTINE_THRESHOLD"]

    # per-check gating: if any hard fail conditions, immediate quarantine
    asr = state["scores"]["asr_confidence"]
    if asr < config["thresholds"]["ASR_CONFIDENCE_FAIL"]:
        decision["action"] = "quarantine"
        decision["reasons"].append("low_asr_confidence")
    # singing spoken overlap hard fail
    singing_info = state.get("singing_removal", {})
    if singing_info.get("spoken_overlap", 0.0) > config["thresholds"]["SINGING_SPOKEN_OVERLAP_FAIL"]:
        decision["action"] = "quarantine"
        decision["reasons"].append("high_spoken_overlap_in_singing_removal")

    # if not hard-failed, use composite thresholds
    if decision["action"] is None:
        if composite >= auto_accept:
            decision["action"] = "finalize"
            decision["reasons"].append("composite_above_auto_accept")
        elif composite < quarantine:
            decision["action"] = "quarantine"
            decision["reasons"].append("composite_below_quarantine")
        else:
            decision["action"] = "finalize_with_flags"
            decision["reasons"].append("composite_between_thresholds")

    # attach decision to state
    state["decision"] = decision

    # write sidecar metadata
    base = state.get("base") or os.path.splitext(os.path.basename(state.get("input_path", "unknown")))[0]
    out_dir = state.get("out_dir") or os.path.dirname(state.get("working_path", ".")) or "."
    sidecar_path = os.path.join(out_dir, f"{base}.metadata.json")

    # build metadata object
    metadata = {
        "pipeline_version": config.get("pipeline_version", "unknown"),
        "input_path": state.get("input_path"),
        "working_path": state.get("working_path"),
        "final_paths": state.get("final_paths", {}),
        "scores": state.get("scores", {}),
        "actions": state.get("actions", []),
        "gpu_log": state.get("gpu_log", []),
        "decision": state["decision"],
        "qa": {"required": False, "notes": []},
        "timestamp": ts
    }

    # if quarantined, mark qa.required true for visibility (but no human signoff required)
    if decision["action"] == "quarantine":
        metadata["qa"]["required"] = True
        metadata["qa"]["notes"].append("auto_quarantine")

    # atomic write
    try:
        write_sidecar_atomic(sidecar_path, metadata)
        log_buffer.append(f"sidecar_written: {sidecar_path}")
    except Exception as e:
        log_buffer.append(f"sidecar_write_error: {e}")

    # final behavior: move or tag files according to action
    # NOTE: actual file moves should be handled by caller; we return metadata and decision
    return state

# -------------------------
# Public API
# -------------------------
def evaluate_and_decide(state, config, log_buffer=None):
    """
    Top-level entry. Computes scores, logs GPU state, applies decision logic,
    and returns updated state. Caller should persist sidecar and move files as needed.
    """
    if log_buffer is None:
        log_buffer = []

    # snapshot GPU state for audit
    gpu_info = get_gpu_memory_info()
    state.setdefault("gpu_log", []).append({"time": now_ts(), "gpus": gpu_info})
    log_buffer.append(f"gpu_snapshot: {gpu_info}")

    # compute composite score
    composite = compute_composite_score(state, config)
    log_buffer.append(f"composite_score: {composite:.3f}")

    # finalize decision and write sidecar
    state = finalize_decision(state, config, log_buffer=log_buffer)
    return state

# -------------------------
# Config loader helper
# -------------------------
def load_config(path):
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # ensure weights exist
    cfg.setdefault("weights", {
        "asr_confidence": 0.30,
        "sermon_confidence": 0.30,
        "singing_quality": 0.20,
        "artifact_score": 0.10,
        "fingerprint_penalty": 0.10
    })
    return cfg
