#!/usr/bin/env python3
"""
reviewer_cli.py

Simple CLI to inspect and act on quarantined files produced by the pipeline.
It updates sidecar JSONs atomically and moves working artifacts between
quarantine and processed directories. Originals are never modified.

Requirements:
  - Python 3.8+
  - PyYAML (pip install pyyaml)
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import yaml

# -------------------------
# Helpers
# -------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    dirpath = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dirpath, delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp = tf.name
    os.replace(tmp, path)

def read_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")
    with open(sidecar_path, "r", encoding="utf8") as f:
        return json.load(f)

def append_audit(sidecar: Dict[str, Any], actor: str, action: str, note: str = "") -> None:
    entry = {"time": time.time(), "actor": actor, "action": action, "note": note}
    sidecar.setdefault("audit", []).append(entry)

def safe_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.move(str(src), str(tmp))
    os.replace(str(tmp), str(dst))

# -------------------------
# CLI operations
# -------------------------
def list_quarantine(cfg: Dict[str, Any]) -> None:
    qdir = Path(cfg["logging"]["quarantine_dir"])
    if not qdir.exists():
        print("Quarantine directory does not exist:", qdir)
        return
    files = sorted(qdir.glob("*.metadata.json"))
    if not files:
        print("No quarantined files found in", qdir)
        return
    print(f"Found {len(files)} quarantined files\n")
    for s in files:
        try:
            meta = read_sidecar(s)
            base = s.stem.replace(".metadata", "")
            score = meta.get("scores", {}).get("composite_score", None)
            decision = meta.get("decision", {}).get("action", "unknown")
            reasons = meta.get("decision", {}).get("reasons", [])
            retries = meta.get("retry_count", 0)
            print(f"- {base}  score={score}  action={decision}  retries={retries}  reasons={reasons}")
        except Exception as e:
            print(f"- {s.name}  ERROR reading sidecar: {e}")

def show_metadata(cfg: Dict[str, Any], base: str) -> None:
    qdir = Path(cfg["logging"]["quarantine_dir"])
    sidecar = qdir / f"{base}.metadata.json"
    if not sidecar.exists():
        print("Sidecar not found in quarantine:", sidecar)
        return
    meta = read_sidecar(sidecar)
    # Pretty print key fields
    print(json.dumps({
        "input_path": meta.get("input_path"),
        "working_path": meta.get("working_path"),
        "final_paths": meta.get("final_paths"),
        "scores": meta.get("scores"),
        "decision": meta.get("decision"),
        "sermon_selection": meta.get("sermon_selection"),
        "singing_removal": meta.get("singing_removal"),
        "gpu_log": meta.get("gpu_log")[-1:] if meta.get("gpu_log") else [],
        "audit_tail": meta.get("audit", [])[-5:]
    }, indent=2))

def accept_file(cfg: Dict[str, Any], base: str, note: str, actor: str = "reviewer_cli") -> None:
    qdir = Path(cfg["logging"]["quarantine_dir"])
    pdir = Path(cfg["logging"]["processed_dir"])
    sidecar = qdir / f"{base}.metadata.json"
    if not sidecar.exists():
        print("Sidecar not found:", sidecar)
        return
    meta = read_sidecar(sidecar)
    # update decision
    meta.setdefault("decision", {})["action"] = "finalize"
    meta.setdefault("decision", {})["reasons"] = meta.get("decision", {}).get("reasons", []) + ["manual_accept"]
    meta["qa"] = {"required": False, "notes": meta.get("qa", {}).get("notes", []) + [note]}
    append_audit(meta, actor, "accept", note)
    # write updated sidecar to processed dir
    out_sidecar = pdir / f"{base}.metadata.json"
    atomic_write_json(out_sidecar, meta)
    # move final artifacts if present
    final_paths = meta.get("final_paths", {})
    for name, path in final_paths.items():
        src = Path(path)
        if src.exists():
            dst = pdir / src.name
            safe_move(src, dst)
            print(f"moved {src} -> {dst}")
    # remove sidecar from quarantine
    try:
        os.remove(sidecar)
    except Exception:
        pass
    print(f"Accepted {base}; sidecar moved to {out_sidecar}")

def reject_file(cfg: Dict[str, Any], base: str, note: str, actor: str = "reviewer_cli") -> None:
    qdir = Path(cfg["logging"]["quarantine_dir"])
    pqdir = Path(cfg["logging"].get("permanent_quarantine_dir", str(qdir / "permanent")))
    sidecar = qdir / f"{base}.metadata.json"
    if not sidecar.exists():
        print("Sidecar not found:", sidecar)
        return
    meta = read_sidecar(sidecar)
    meta.setdefault("decision", {})["action"] = "permanent_quarantine"
    meta.setdefault("decision", {})["reasons"] = meta.get("decision", {}).get("reasons", []) + ["manual_reject"]
    meta["qa"] = {"required": True, "notes": meta.get("qa", {}).get("notes", []) + [note]}
    append_audit(meta, actor, "reject", note)
    out_sidecar = pqdir / f"{base}.metadata.json"
    atomic_write_json(out_sidecar, meta)
    # move working artifacts if present
    working = Path(meta.get("working_path", ""))
    if working.exists():
        dst = pqdir / working.name
        safe_move(working, dst)
        print(f"moved working artifact {working} -> {dst}")
    # remove original quarantine sidecar
    try:
        os.remove(sidecar)
    except Exception:
        pass
    print(f"Rejected {base}; moved to permanent quarantine {pqdir}")

def requeue_file(cfg: Dict[str, Any], base: str, note: str, actor: str = "reviewer_cli") -> None:
    qdir = Path(cfg["logging"]["quarantine_dir"])
    watch_dir = Path(cfg.get("watch_dir", cfg["logging"].get("sidecar_dir", ".")))
    sidecar = qdir / f"{base}.metadata.json"
    if not sidecar.exists():
        print("Sidecar not found:", sidecar)
        return
    meta = read_sidecar(sidecar)
    # reset retry counters and last_error
    meta["retry_count"] = 0
    meta.pop("last_error", None)
    append_audit(meta, actor, "requeue", note)
    # write updated sidecar back to watch dir so orchestrator picks it up
    out_sidecar = watch_dir / f"{base}.metadata.json"
    atomic_write_json(out_sidecar, meta)
    # move working artifact back to watch dir if present
    working = Path(meta.get("working_path", ""))
    if working.exists():
        dst = watch_dir / working.name
        safe_move(working, dst)
        print(f"moved working artifact {working} -> {dst}")
    # remove quarantine sidecar
    try:
        os.remove(sidecar)
    except Exception:
        pass
    print(f"Requeued {base} for automated reprocessing")

# -------------------------
# CLI entrypoint
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Reviewer CLI for quarantined files")
    p.add_argument("--config", required=True, help="Path to production_config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List quarantined files")

    show = sub.add_parser("show", help="Show metadata for a quarantined file")
    show.add_argument("base", help="Base filename (without extension)")

    accept = sub.add_parser("accept", help="Accept a quarantined file and move to processed")
    accept.add_argument("base", help="Base filename")
    accept.add_argument("--note", default="accepted", help="Short note to append to audit")

    reject = sub.add_parser("reject", help="Reject a quarantined file permanently")
    reject.add_argument("base", help="Base filename")
    reject.add_argument("--note", default="rejected", help="Short note to append to audit")

    requeue = sub.add_parser("requeue", help="Requeue a quarantined file for automated reprocessing")
    requeue.add_argument("base", help="Base filename")
    requeue.add_argument("--note", default="requeue", help="Short note to append to audit")

    args = p.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "list":
        list_quarantine(cfg)
    elif args.cmd == "show":
        show_metadata(cfg, args.base)
    elif args.cmd == "accept":
        accept_file(cfg, args.base, args.note)
    elif args.cmd == "reject":
        reject_file(cfg, args.base, args.note)
    elif args.cmd == "requeue":
        requeue_file(cfg, args.base, args.note)
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()
