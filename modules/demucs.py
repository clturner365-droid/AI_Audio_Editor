# modules/demucs_v2.py
# GPU‑pinned Demucs wrapper for optional speech enhancement

import torch
from modules.logging_system import append_file_log

_DEMUCS_MODEL = None


def load_demucs(log_buffer):
    global _DEMUCS_MODEL

    if _DEMUCS_MODEL is not None:
        return _DEMUCS_MODEL

    append_file_log(log_buffer, "v2: Loading Demucs model on GPU 1...")

    try:
        device = torch.device("cuda:1")  # PINNED TO GPU 1
        from demucs.pretrained import get_model

        _DEMUCS_MODEL = get_model("htdemucs").to(device)
        append_file_log(log_buffer, "v2: Demucs loaded on GPU 1.")

    except Exception as e:
        append_file_log(log_buffer, f"v2: ERROR loading Demucs: {e}")
        _DEMUCS_MODEL = None

    return _DEMUCS_MODEL
