#!/usr/bin/env python3
"""
modules/transcript.py

ASR wrapper used by the pipeline.

Function:
    generate_transcript(waveform, sr, log_buffer, *, device="cuda:0", diarize=False, speaker_enrollments=None)

Returns:
    transcript_text: str
    segments: List[dict]
    confidences: List[float]
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import tempfile
import os
import time
import json
import subprocess

# Optional ASR backends
try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None

# Optional diarization
try:
    from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore
except Exception:
    PyannotePipeline = None

from modules.logging_system import append_file_log
from modules.stepwise_saving import maybe_save_step_audio


# ----------------------------------------------------------------------
# Helper: write waveform to temp WAV
# ----------------------------------------------------------------------

def _write_temp_wav(waveform: np.ndarray, sr: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    import soundfile as sf
    sf.write(path, waveform, sr, subtype="PCM_16")
    return path


def _safe_mean_confidence(segment_confidences: List[float]) -> float:
    if not segment_confidences:
        return 0.0
    return float(sum(segment_confidences) / len(segment_confidences))


def _normalize_confidence(raw_conf: float) -> float:
    try:
        if raw_conf is None:
            return 0.0
        if raw_conf > 1.5:
            return max(0.0, min(1.0, raw_conf / 100.0))
        return max(0.0, min(1.0, float(raw_conf)))
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Faster-Whisper backend
# ----------------------------------------------------------------------

def _run_faster_whisper(wav_path: str, device: str, log_buffer: list, model_name: str = "small") -> Tuple[str, List[Dict[str, Any]], List[float]]:
    if WhisperModel is None:
        log_buffer.append("faster-whisper not available in environment.")
        return "", [], []

    try:
        log_buffer.append(f"faster-whisper: loading model {model_name} on {device}")
        model = WhisperModel(model_name, device=device, compute_type="float16")

        segments_iter, info = model.transcribe(wav_path, beam_size=5, vad_filter=True)
        segments = []
        confidences = []
        texts = []

        for seg in segments_iter:
            start = float(seg.start)
            end = float(seg.end)
            text = seg.text.strip()

            raw_conf = getattr(seg, "avg_logprob", None)
            conf = 0.0
            if raw_conf is not None:
                try:
                    conf = 1.0 / (1.0 + np.exp(-raw_conf))
                except Exception:
                    conf = 0.0

            segments.append({"start": start, "end": end, "text": text})
            confidences.append(_normalize_confidence(conf))
            texts.append(text)

        full_text = " ".join(texts)
        log_buffer.append(f"faster-whisper: transcribed {len(segments)} segments")
        return full_text, segments, confidences

    except Exception as e:
        log_buffer.append(f"faster-whisper error: {e}")
        return "", [], []


# ----------------------------------------------------------------------
# Whisper fallback
# ----------------------------------------------------------------------

def _run_whisper_fallback(wav_path: str, log_buffer: list) -> Tuple[str, List[Dict[str, Any]], List[float]]:
    try:
        import soundfile as sf
        wav, sr = sf.read(wav_path, dtype="float32")
        duration = len(wav) / float(sr)
        log_buffer.append("whisper_fallback: no ASR backend available; returning empty transcript placeholder")
        return "", [{"start": 0.0, "end": duration, "text": ""}], [0.0]
    except Exception as e:
        log_buffer.append(f"whisper_fallback error: {e}")
        return "", [], []


# ----------------------------------------------------------------------
# Pyannote diarization
# ----------------------------------------------------------------------

def _run_pyannote_diarization(wav_path: str, device: str, log_buffer: list) -> Optional[List[Dict[str, Any]]]:
    if PyannotePipeline is None:
        log_buffer.append("pyannote not available; skipping diarization")
        return None

    try:
        log_buffer.append("pyannote: running diarization pipeline")
        pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=None)
        diarization = pipeline(wav_path)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})

        log_buffer.append(f"pyannote: diarization produced {len(segments)} segments")
        return segments

    except Exception as e:
        log_buffer.append(f"pyannote diarization error: {e}")
        return None


# ----------------------------------------------------------------------
# Merge ASR + diarization
# ----------------------------------------------------------------------

def _merge_asr_and_diarization(asr_segments: List[Dict[str, Any]], diarization_segments: Optional[List[Dict[str, Any]]], log_buffer: list) -> List[Dict[str, Any]]:
    if not diarization_segments:
        return asr_segments

    diar = diarization_segments
    out = []

    for seg in asr_segments:
        s0, s1 = seg.get("start", 0.0), seg.get("end", 0.0)
        assigned = None
        max_overlap = 0.0

        for d in diar:
            d0, d1 = d.get("start", 0.0), d.get("end", 0.0)
            overlap = max(0.0, min(s1, d1) - max(s0, d0))
            if overlap > max_overlap:
                max_overlap = overlap
                assigned = d.get("speaker")

        if assigned is not None and max_overlap > 0.0:
            seg["speaker"] = assigned

        out.append(seg)

    return out


# ----------------------------------------------------------------------
# Top-level ASR function (unchanged)
# ----------------------------------------------------------------------

def generate_transcript(waveform: np.ndarray, sr: int, log_buffer: list, *, device: str = "cuda:0", diarize: bool = False, speaker_enrollments: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], List[float]]:
    wav_path = _write_temp_wav(waveform, sr)

    try:
        if WhisperModel is not None:
            full_text, asr_segments, confidences = _run_faster_whisper(wav_path, device, log_buffer)
        else:
            full_text, asr_segments, confidences = _run_whisper_fallback(wav_path, log_buffer)

        diarization_segments = None
        if diarize:
            diarization_segments = _run_pyannote_diarization(wav_path, device, log_buffer)

        merged_segments = _merge_asr_and_diarization(asr_segments, diarization_segments, log_buffer)

        if not confidences or len(confidences) != len(merged_segments):
            confidences = [0.8 if seg.get("text", "").strip() else 0.0 for seg in merged_segments]

        confidences = [_normalize_confidence(c) for c in confidences]

        if not full_text:
            full_text = " ".join([s.get("text", "") for s in merged_segments]).strip()

        log_buffer.append(f"generate_transcript: returning {len(merged_segments)} segments, transcript length {len(full_text)}")
        return full_text, merged_segments, confidences

    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Dispatcher wrapper
# ----------------------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher entry point for ASR transcription.

    This wrapper:
      - pulls waveform and sr from state
      - reads device + diarization flags from ctx
      - calls generate_transcript()
      - updates state["transcript"], state["segments"], state["transcript_confidences"]
      - logs the action
      - triggers stepwise save if enabled
    """

    log_buffer = ctx["log_buffer"]
    device = ctx.get("gpu_for_transcribe", "cuda:0")
    diarize = bool(ctx.get("enable_diarization", False))
    save_stepwise = bool(ctx.get("save_stepwise", False))

    append_file_log(log_buffer, "=== Step: generate_transcript ===")

    wav = state.get("waveform")
    sr = state.get("sr")

    if wav is None or sr is None:
        append_file_log(log_buffer, "No waveform or sample rate in state; skipping transcription.")
        return state

    transcript, segments, confidences = generate_transcript(
        wav,
        sr,
        log_buffer,
        device=device,
        diarize=diarize
    )

    state["transcript"] = transcript
    state["segments"] = segments
    state["transcript_confidences"] = confidences

    state.setdefault("actions", []).append({
        "step": "generate_transcript",
        "time": time.time(),
        "segments": len(segments),
        "chars": len(transcript)
    })

    if save_stepwise:
        maybe_save_step_audio("generate_transcript", state, ctx)

    return state

