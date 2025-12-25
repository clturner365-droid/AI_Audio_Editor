# modules/intro_outro_removal.py
# Detects and removes old intros/outros using silence and energy analysis.

import numpy as np
import librosa
from modules.logging_system import append_file_log


# Tunable constants – we can adjust later if needed
TARGET_SAMPLE_RATE = 16000          # should match audio_loader
MAX_INTRO_SECONDS = 25              # don't trim more than this from start
MAX_OUTRO_SECONDS = 35              # don't trim more than this from end
MIN_SPEECH_BLOCK_SECONDS = 20       # minimum "real sermon" chunk length
SILENCE_TOP_DB = 35                 # for librosa.effects.split


def _rms(signal: np.ndarray) -> float:
    """Root mean square energy."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def _describe_segment(label: str, segment: np.ndarray, sr: int, log_buffer):
    """Log some basic stats for a segment."""
    duration = len(segment) / sr
    energy = _rms(segment)
    append_file_log(
        log_buffer,
        f"{label}: duration={duration:.2f}s, rms={energy:.6f}"
    )


def remove_intros_outros(waveform: np.ndarray, sr: int, log_buffer):
    """
    Remove old intros/outros using:
      - silence-based splitting
      - energy and duration heuristics

    Strategy:
      1. Split audio into non-silent segments.
      2. Identify the main sermon chunk (longest, highest energy).
      3. Trim everything before it as potential intro.
      4. Trim everything after it as potential outro (within limits).
    """

    append_file_log(log_buffer, "Starting intro/outro detection...")

    if len(waveform) == 0:
        append_file_log(log_buffer, "Waveform empty; skipping intro/outro removal.")
        return waveform

    # Ensure mono, float32
    waveform = waveform.astype(np.float32).flatten()

    # 1. Split into non-silent chunks (librosa does the heavy lifting)
    non_silent_intervals = librosa.effects.split(
        waveform,
        top_db=SILENCE_TOP_DB
    )

    if len(non_silent_intervals) == 0:
        append_file_log(log_buffer, "No non-silent regions found; returning original waveform.")
        return waveform

    append_file_log(log_buffer, f"Found {len(non_silent_intervals)} non-silent segments.")

    # 2. Analyze each segment: duration and energy
    segments = []
    for idx, (start, end) in enumerate(non_silent_intervals):
        seg = waveform[start:end]
        duration = (end - start) / sr
        energy = _rms(seg)
        segments.append({
            "index": idx,
            "start": start,
            "end": end,
            "duration": duration,
            "energy": energy,
        })
        append_file_log(
            log_buffer,
            f"Segment {idx}: start={start}, end={end}, duration={duration:.2f}s, rms={energy:.6f}"
        )

    # 3. Heuristic: main sermon = longest reasonably loud segment
    #    (we prioritize duration; energy is secondary)
    segments_sorted = sorted(
        segments,
        key=lambda s: (s["duration"], s["energy"]),
        reverse=True
    )
    main_seg = segments_sorted[0]
    main_start = main_seg["start"]
    main_end = main_seg["end"]

    _describe_segment("Chosen main sermon segment", waveform[main_start:main_end], sr, log_buffer)

    # 4. Determine intro trim point (before main sermon)
    intro_trim_samples = 0
    max_intro_samples = int(MAX_INTRO_SECONDS * sr)

    if main_start > 0:
        # Only allow trimming up to MAX_INTRO_SECONDS
        intro_trim_samples = min(main_start, max_intro_samples)
        intro_seconds = intro_trim_samples / sr
        append_file_log(
            log_buffer,
            f"Intro candidate: trimming {intro_trim_samples} samples ({intro_seconds:.2f}s) before main sermon."
        )
    else:
        append_file_log(log_buffer, "No intro detected before main sermon.")

    # 5. Determine outro trim point (after main sermon)
    total_len = len(waveform)
    outro_trim_start = total_len
    max_outro_samples = int(MAX_OUTRO_SECONDS * sr)

    if main_end < total_len:
        # Everything after main_end is candidate outro, but clamp to MAX_OUTRO_SECONDS
        outro_region_len = total_len - main_end
        # If outro region is longer than allowed, keep some tail to avoid over-trim
        if outro_region_len > max_outro_samples:
            outro_trim_start = main_end + (outro_region_len - max_outro_samples)
        else:
            outro_trim_start = main_end

        outro_seconds = (total_len - outro_trim_start) / sr
        append_file_log(
            log_buffer,
            f"Outro candidate: trimming from sample {outro_trim_start}, about {outro_seconds:.2f}s."
        )
    else:
        append_file_log(log_buffer, "No outro detected after main sermon.")

    # 6. Safety: ensure remaining core is not too short
    cleaned = waveform[intro_trim_samples:outro_trim_start]
    cleaned_duration = len(cleaned) / sr

    if cleaned_duration < MIN_SPEECH_BLOCK_SECONDS:
        append_file_log(
            log_buffer,
            f"Cleaned segment too short ({cleaned_duration:.2f}s). "
            f"Reverting to original waveform with no intro/outro removal."
        )
        return waveform

    append_file_log(
        log_buffer,
        f"Intro/outro removal applied. Final duration: {cleaned_duration:.2f}s "
        f"({len(cleaned)} samples)."
    )

    return cleaned
