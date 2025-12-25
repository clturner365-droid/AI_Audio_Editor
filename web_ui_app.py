#!/usr/bin/env python3
"""
Simple Web UI for quarantined files.
Run: FLASK_APP=app.py flask run --host=0.0.0.0 --port=8080
"""

import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template, abort

import yaml

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load config
CFG_PATH = os.environ.get("SERMON_CONFIG", "production_config.yaml")
with open(CFG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

QDIR = Path(CONFIG["logging"]["quarantine_dir"])
PDIR = Path(CONFIG["logging"]["processed_dir"])
WATCH_DIR = Path(CONFIG.get("watch_dir", CONFIG["logging"].get("sidecar_dir", ".")))
PQDIR = Path(CONFIG["logging"].get("permanent_quarantine_dir", str(QDIR / "permanent")))

# Helpers
def read_sidecar(path: Path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as tf:
        json.dump(data, tf, indent=2)
        tmp = Path(tf.name)
    os.replace(str(tmp), str(path))

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.move(str(src), str(tmp))
    os.replace(str(tmp), str(dst))

def list_quarantined():
    files = sorted(QDIR.glob("*.metadata.json"))
    out = []
    for p in files:
        try:
            meta = read_sidecar(p)
            base = p.stem.replace(".metadata", "")
            out.append({
                "base": base,
                "score": meta.get("scores", {}).get("composite_score"),
                "decision": meta.get("decision", {}).get("action"),
                "reasons": meta.get("decision", {}).get("reasons", []),
                "retry_count": meta.get("retry_count", 0),
                "sidecar": str(p)
            })
        except Exception:
            continue
    return out

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/list")
def api_list():
    return jsonify(list_quarantined())

@app.route("/api/show/<base>")
def api_show(base):
    sidecar = QDIR / f"{base}.metadata.json"
    if not sidecar.exists():
        return jsonify({"error": "not_found"}), 404
    meta = read_sidecar(sidecar)
    return jsonify(meta)

@app.route("/api/audio/<base>/<kind>")
def api_audio(base, kind):
    # kind: working | final | no_singing | sermon
    sidecar = QDIR / f"{base}.metadata.json"
    if not sidecar.exists():
        return jsonify({"error": "not_found"}), 404
    meta = read_sidecar(sidecar)
    final_paths = meta.get("final_paths", {})
    path = None
    if kind == "working":
        path = Path(meta.get("working_path", ""))
    else:
        path = Path(final_paths.get(kind, ""))
    if not path or not path.exists():
        return jsonify({"error": "audio_not_found"}), 404
    return send_file(str(path), conditional=True)

@app.route("/api/action", methods=["POST"])
def api_action():
    """
    JSON body: {"action":"accept"|"reject"|"requeue", "base":"...", "note":"..."}
    """
    body = request.get_json(force=True)
    action = body.get("action")
    base = body.get("base")
    note = body.get("note", "")
    actor = body.get("actor", "web_ui")
    sidecar = QDIR / f"{base}.metadata.json"
    if not sidecar.exists():
        return jsonify({"error": "not_found"}), 404
    meta = read_sidecar(sidecar)

    if action == "accept":
        meta.setdefault("decision", {})["action"] = "finalize"
        meta.setdefault("decision", {})["reasons"] = meta.get("decision", {}).get("reasons", []) + ["web_accept"]
        meta["qa"] = {"required": False, "notes": meta.get("qa", {}).get("notes", []) + [note]}
        meta.setdefault("audit", []).append({"time": time.time(), "actor": actor, "action": "accept", "note": note})
        # write sidecar to processed dir and move artifacts
        out_sidecar = PDIR / f"{base}.metadata.json"
        atomic_write_json(out_sidecar, meta)
        for name, path in meta.get("final_paths", {}).items():
            src = Path(path)
            if src.exists():
                dst = PDIR / src.name
                safe_move(src, dst)
        try:
            os.remove(sidecar)
        except Exception:
            pass
        return jsonify({"result": "accepted"})

    if action == "reject":
        meta.setdefault("decision", {})["action"] = "permanent_quarantine"
        meta.setdefault("decision", {})["reasons"] = meta.get("decision", {}).get("reasons", []) + ["web_reject"]
        meta["qa"] = {"required": True, "notes": meta.get("qa", {}).get("notes", []) + [note]}
        meta.setdefault("audit", []).append({"time": time.time(), "actor": actor, "action": "reject", "note": note})
        out_sidecar = PQDIR / f"{base}.metadata.json"
        atomic_write_json(out_sidecar, meta)
        # move working artifact if present
        working = Path(meta.get("working_path", ""))
        if working.exists():
            dst = PQDIR / working.name
            safe_move(working, dst)
        try:
            os.remove(sidecar)
        except Exception:
            pass
        return jsonify({"result": "rejected"})

    if action == "requeue":
        meta["retry_count"] = 0
        meta.pop("last_error", None)
        meta.setdefault("audit", []).append({"time": time.time(), "actor": actor, "action": "requeue", "note": note})
        out_sidecar = WATCH_DIR / f"{base}.metadata.json"
        atomic_write_json(out_sidecar, meta)
        working = Path(meta.get("working_path", ""))
        if working.exists():
            dst = WATCH_DIR / working.name
            safe_move(working, dst)
        try:
            os.remove(sidecar)
        except Exception:
            pass
        return jsonify({"result": "requeued"})

    return jsonify({"error": "unknown_action"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
