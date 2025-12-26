#!/usr/bin/env python3
"""
modules/transcript.py

ASR wrapper used by the pipeline.

Function:
    generate_transcript(waveform, sr, log_buffer, *, device="cuda:0", diarize=False, speaker_enrollments=None)

Returns:
    transcript_text: str
    segments: List[dict]  # each dict: {"start": float, "end": float, "text": str, "speaker": optional}
    confidences: List[float]  # per-segment mean confidence (0..1)
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

# Helper: write waveform to temp WAV for model APIs that accept file paths
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
    # If backend returns 0..100 or -inf, normalize to 0..1
    try:
        if raw_conf is None:
            return 0.0
        if raw_conf > 1.5:  # likely 0..100
            return max(0.0, min(1.0, raw_conf / 100.0))
        return max(0.0, min(1.0, float(raw_conf)))
    except Exception:
        return 0.0

def _run_faster_whisper(wav_path: str, device: str, log_buffer: list, model_name: str = "small") -> Tuple[str, List[Dict[str, Any]], List[float]]:
    """
    Uses faster-whisper WhisperModel.transcribe to produce segments.
    Returns (full_text, segments, confidences)
    """
    if WhisperModel is None:
        log_buffer.append("faster-whisper not available in environment.")
        return "", [], []

    try:
        log_buffer.append(f"faster-whisper: loading model {model_name} on {device}")
        model = WhisperModel(model_name, device=device, compute_type="float16")
        # beam_size and other params can be tuned
        segments_iter, info = model.transcribe(wav_path, beam_size=5, vad_filter=True)
        segments = []
        confidences = []
        texts = []
        for seg in segments_iter:
            start = float(seg.start)
            end = float(seg.end)
            text = seg.text.strip()
            # faster-whisper segment may include 'avg_logprob' or 'no_speech_prob' in seg.__dict__
            raw_conf = getattr(seg, "avg_logprob", None)
            # convert logprob to a 0..1 proxy if present (this is heuristic)
            conf = 0.0
            if raw_conf is not None:
                try:
                    # avg_logprob typically negative; map to 0..1 via sigmoid-like transform
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

def _run_whisper_fallback(wav_path: str, log_buffer: list) -> Tuple[str, List[Dict[str, Any]], List[float]]:
    """
    Minimal fallback that calls whisper.cpp or returns empty transcript.
    This placeholder returns a single segment covering the whole file with empty text.
    """
    try:
        import soundfile as sf
        wav, sr = sf.read(wav_path, dtype="float32")
        duration = len(wav) / float(sr)
        log_buffer.append("whisper_fallback: no ASR backend available; returning empty transcript placeholder")
        return "", [{"start": 0.0, "end": duration, "text": ""}], [0.0]
    except Exception as e:
        log_buffer.append(f"whisper_fallback error: {e}")
        return "", [], []

def _run_pyannote_diarization(wav_path: str, device: str, log_buffer: list) -> Optional[List[Dict[str, Any]]]:
    """
    If pyannote Pipeline is available and configured via env, run diarization and return
    a list of speaker segments: {"start":, "end":, "speaker": "spk_1"}.
    This function expects a pyannote pipeline token configured in the environment if required.
    """
    if PyannotePipeline is None:
        log_buffer.append("pyannote not available; skipping diarization")
        return None
    try:
        log_buffer.append("pyannote: running diarization pipeline")
        # The pipeline string depends on your pyannote setup; using "pyannote/speaker-diarization" as placeholder
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

def _merge_asr_and_diarization(asr_segments: List[Dict[str, Any]], diarization_segments: Optional[List[Dict[str, Any]]], log_buffer: list) -> List[Dict[str, Any]]:
    """
    If diarization segments are available, attempt to assign speaker labels to ASR segments
    by overlap. Returns ASR segments augmented with 'speaker' when matched.
    """
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

def generate_transcript(waveform: np.ndarray, sr: int, log_buffer: list, *, device: str = "cuda:0", diarize: bool = False, speaker_enrollments: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], List[float]]:
    """
    Top-level ASR function used by the pipeline.

    Parameters:
      waveform: numpy array (mono float32)
      sr: sample rate
      log_buffer: list to append log lines
      device: device string for ASR (e.g., 'cuda:0' or 'cpu')
      diarize: whether to run diarization and attach speaker labels
      speaker_enrollments: optional path to enrollment registry for speaker verification (not used here)

    Returns:
      (transcript_text, segments, confidences)
    """
    # write temp WAV for model backends that require a file
    wav_path = _write_temp_wav(waveform, sr)
    try:
        # Try faster-whisper first
        if WhisperModel is not None:
            full_text, asr_segments, confidences = _run_faster_whisper(wav_path, device, log_buffer)
        else:
            full_text, asr_segments, confidences = _run_whisper_fallback(wav_path, log_buffer)

        # Optionally run diarization and merge speaker labels
        diarization_segments = None
        if diarize:
            diarization_segments = _run_pyannote_diarization(wav_path, device, log_buffer)

        merged_segments = _merge_asr_and_diarization(asr_segments, diarization_segments, log_buffer)

        # Ensure confidences list aligns with segments
        if not confidences or len(confidences) != len(merged_segments):
            # fallback: assign uniform confidences based on presence of text
            confidences = []
            for seg in merged_segments:
                txt = seg.get("text", "")
                confidences.append(0.8 if txt.strip() else 0.0)

        # Normalize confidences to 0..1
        confidences = [_normalize_confidence(c) for c in confidences]

        # Final full_text if empty: join segment texts
        if not full_text:
            full_text = " ".join([s.get("text", "") for s in merged_segments]).strip()

        log_buffer.append(f"generate_transcript: returning {len(merged_segments)} segments, transcript length {len(full_text)}")
        return full_text, merged_segments, confidences
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass
