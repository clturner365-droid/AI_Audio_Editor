# intro_outro.py
# Handles intro/outro selection, generation, loudness matching,
# multi-speaker logic, and final audio assembly.

import os
import numpy as np

from modules.logging_system import append_file_log
from modules.registry import (
    get_next_intro_index,
    get_next_outro_index,
    update_intro_rotation,
    update_outro_rotation
)

# Base folders
INTRO_DIR = os.path.join("PipelineB", "Intros")
OUTRO_DIR = os.path.join("PipelineB", "Outros")

MULTI_INTRO_DIR = os.path.join(INTRO_DIR, "_multi")
MULTI_OUTRO_DIR = os.path.join(OUTRO_DIR, "_multi")


# ---------------------------------------------------------
# DIRECTORY SETUP
# ---------------------------------------------------------

def ensure_intro_outro_dirs():
    """
    Ensures all required directories exist.
    """
    os.makedirs(INTRO_DIR, exist_ok=True)
    os.makedirs(OUTRO_DIR, exist_ok=True)
    os.makedirs(MULTI_INTRO_DIR, exist_ok=True)
    os.makedirs(MULTI_OUTRO_DIR, exist_ok=True)


# ---------------------------------------------------------
# LOADING INTRO/OUTRO FILES
# ---------------------------------------------------------

def load_audio_file(path):
    """
    Placeholder for loading intro/outro WAV files.
    """
    # TODO: Replace with real audio loading
    return np.zeros(1, dtype=np.float32), 16000


def load_intro(canonical_name, index, log_buffer):
    """
    Loads the selected intro for a speaker.
    """
    path = os.path.join(INTRO_DIR, canonical_name, f"intro_{index}.wav")

    if not os.path.exists(path):
        append_file_log(log_buffer, f"Missing intro file: {path}")
        return None, None

    return load_audio_file(path)


def load_outro(canonical_name, index, log_buffer):
    """
    Loads the selected outro for a speaker.
    """
    path = os.path.join(OUTRO_DIR, canonical_name, f"outro_{index}.wav")

    if not os.path.exists(path):
        append_file_log(log_buffer, f"Missing outro file: {path}")
        return None, None

    return load_audio_file(path)


# ---------------------------------------------------------
# GENERATING NEW INTROS/OUTROS
# ---------------------------------------------------------

def generate_intro(canonical_name, index, log_buffer):
    """
    Generates a new intro using a male voice model.
    Placeholder for TTS.
    """
    append_file_log(log_buffer, f"Generating intro {index} for {canonical_name}...")

    # TODO: Replace with real TTS
    waveform = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds placeholder
    sample_rate = 16000

    out_dir = os.path.join(INTRO_DIR, canonical_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"intro_{index}.wav")
    # TODO: Save waveform to disk

    return waveform, sample_rate


def generate_outro(canonical_name, index, log_buffer):
    """
    Generates a new outro using a male voice model.
    """
    append_file_log(log_buffer, f"Generating outro {index} for {canonical_name}...")

    # TODO: Replace with real TTS
    waveform = np.zeros(16000 * 3, dtype=np.float32)
    sample_rate = 16000

    out_dir = os.path.join(OUTRO_DIR, canonical_name)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"outro_{index}.wav")
    # TODO: Save waveform to disk

    return waveform, sample_rate


# ---------------------------------------------------------
# LOUDNESS MATCHING
# ---------------------------------------------------------

def match_loudness(intro_wave, sermon_wave, sample_rate):
    """
    Adjusts intro/outro loudness to match sermon loudness.
    Placeholder for LUFS matching.
    """
    # TODO: Implement LUFS matching
    return intro_wave


# ---------------------------------------------------------
# MULTI-SPEAKER GENERIC INTRO/OUTRO
# ---------------------------------------------------------

def load_multi_intro(log_buffer):
    """
    Loads the generic multi-speaker intro.
    """
    path = os.path.join(MULTI_INTRO_DIR, "intro_generic.wav")

    if not os.path.exists(path):
        append_file_log(log_buffer, "Missing generic multi-speaker intro.")
        return None, None

    return load_audio_file(path)


def load_multi_outro(log_buffer):
    """
    Loads the generic multi-speaker outro.
    """
    path = os.path.join(MULTI_OUTRO_DIR, "outro_generic.wav")

    if not os.path.exists(path):
        append_file_log(log_buffer, "Missing generic multi-speaker outro.")
        return None, None

    return load_audio_file(path)


# ---------------------------------------------------------
# FINAL ASSEMBLY
# ---------------------------------------------------------

def add_silence(seconds, sample_rate):
    """
    Returns a waveform of digital silence.
    """
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def assemble_final_audio(intro, sermon, outro, sample_rate):
    """
    Concatenates intro + sermon + outro + silence.
    """
    silence = add_silence(10, sample_rate)
    parts = [p for p in [intro, sermon, outro, silence] if p is not None]
    return np.concatenate(parts)


# ---------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------------------------

def apply_intro_outro(registry, canonical_names, sermon_wave, sample_rate, multi, log_buffer):
    """
    Full intro/outro workflow:
        - Multi-speaker → generic intro/outro
        - Single speaker → rotated intro/outro
        - Loudness match
        - Add 10 seconds silence
        - Return final waveform
    """

    if multi:
        append_file_log(log_buffer, "MULTI-SPEAKER: Using generic intro/outro.")

        intro, _ = load_multi_intro(log_buffer)
        outro, _ = load_multi_outro(log_buffer)

        if intro is not None:
            intro = match_loudness(intro, sermon_wave, sample_rate)

        if outro is not None:
            outro = match_loudness(outro, sermon_wave, sample_rate)

        final = assemble_final_audio(intro, sermon_wave, outro, sample_rate)
        return final

    # SINGLE SPEAKER
    canonical = canonical_names[0]
    entry = registry.get(canonical)

    # Determine next intro/outro in rotation
    intro_index = get_next_intro_index(entry)
    outro_index = get_next_outro_index(entry)

    # Load or generate intro
    intro, _ = load_intro(canonical, intro_index, log_buffer)
    if intro is None:
        intro, _ = generate_intro(canonical, intro_index, log_buffer)

    # Load or generate outro
    outro, _ = load_outro(canonical, outro_index, log_buffer)
    if outro is None:
        outro, _ = generate_outro(canonical, outro_index, log_buffer)

    # Loudness match
    intro = match_loudness(intro, sermon_wave, sample_rate)
    outro = match_loudness(outro, sermon_wave, sample_rate)

    # Update rotation counters
    update_intro_rotation(entry, intro_index)
    update_outro_rotation(entry, outro_index)

    # Assemble final audio
    final = assemble_final_audio(intro, sermon_wave, outro, sample_rate)
    return final
