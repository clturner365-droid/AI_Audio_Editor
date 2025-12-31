# modules/vad_engine_v2.py
# GPU‑pinned pyannote segmentation model (VAD)

import numpy as np
from modules.logging_system import append_file_log

try:
    import torch
    from pyannote.audio import Model
except ImportError:
    torch = None
    Model = None

_VAD_MODEL = None
VAD_MODEL_NAME = "pyannote/segmentation"


def load_vad_model(log_buffer):
    global _VAD_MODEL, Model, torch

    if _VAD_MODEL is not None:
        return _VAD_MODEL

    if Model is None or torch is None:
        append_file_log(log_buffer, "v2: pyannote/torch unavailable; VAD disabled.")
        return None

    append_file_log(log_buffer, "v2: Loading pyannote VAD model on GPU 0...")

    try:
        device = torch.device("cuda:0")  # PINNED TO GPU 0
        _VAD_MODEL = Model.from_pretrained(
            VAD_MODEL_NAME,
            use_auth_token=None
        ).to(device)

        append_file_log(log_buffer, "v2: VAD model loaded on GPU 0.")

    except Exception as e:
        append_file_log(log_buffer, f"v2: ERROR loading VAD model: {e}")
        _VAD_MODEL = None

    return _VAD_MODEL
