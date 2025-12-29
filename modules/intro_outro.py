#!/usr/bin/env python3
"""
modules/intro_outro.py

Generates intro + outro using Coqui TTS (same voice for both),
normalizes loudness, and assembles final audio using pydub.
"""

import os
from pydub import AudioSegment

from modules.logging_system import append_file_log
from modules.tts_audio_generator import generate_tts_audio


# ---------------------------------------------------------
# TEXT GENERATION
# ---------------------------------------------------------

def build_intro_text(state):
    """
    Build intro text based on metadata.
    Customize as needed.
    """
    speaker = state.get("speaker_name", "your pastor")
    title = state.get("sermon_title", "today's message")
    return f"You are listening to {speaker} with {title}."


def build_outro_text(state):
    """
    Build outro text based on metadata.
    Customize as needed.
    """
    return "Thank you for listening. Please join us again next time."


# ---------------------------------------------------------
# LOUDNESS MATCHING (placeholder)
# ---------------------------------------------------------

def match_loudness(target: AudioSegment, reference: AudioSegment):
    """
    Placeholder for LUFS matching.
    For now, return target unchanged.
    """
    return target


# ---------------------------------------------------------
# FINAL ASSEMBLY
# ---------------------------------------------------------

def add_silence(seconds: float, sample_rate: int):
    """
    Returns pydub silence.
    """
    return AudioSegment.silent(duration=int(seconds * 1000), frame_rate=sample_rate)


def assemble_final_audio(intro: AudioSegment,
                         sermon: AudioSegment,
                         outro: AudioSegment,
                         sample_rate: int):
    """
    Concatenate intro + sermon + outro + 10 seconds silence.
    """
    silence = add_silence(10, sample_rate)
    return intro + sermon + outro + silence


# ---------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------------------------

def apply_intro_outro(state, ctx):
    """
    Full intro/outro workflow:

        - Use TTS to generate intro/outro
        - Use SAME voice for both (state["tts_voice"])
        - Loudness match
        - Add 10 seconds silence
        - Return final AudioSegment
    """

    log_buffer = ctx["log_buffer"]
    append_file_log(log_buffer, "intro_outro: starting")

    # Ensure voice is selected by dispatcher
    voice = state.get("tts_voice")
    if not voice:
        append_file_log(log_buffer, "intro_outro: no tts_voice in state; skipping")
        return state

    # Load sermon audio (AudioSegment)
    sermon_path = state.get("working_path")
    if not sermon_path or not os.path.exists(sermon_path):
        append_file_log(log_buffer, "intro_outro: missing sermon audio")
        return state

    sermon_audio = AudioSegment.from_wav(sermon_path)
    sample_rate = sermon_audio.frame_rate

    # Build texts
    intro_text = build_intro_text(state)
    outro_text = build_outro_text(state)

    # Generate TTS audio
    intro_audio = generate_tts_audio(intro_text, speaker=voice, log_buffer=log_buffer)
    outro_audio = generate_tts_audio(outro_text, speaker=voice, log_buffer=log_buffer)

    # Loudness match
    intro_audio = match_loudness(intro_audio, sermon_audio)
    outro_audio = match_loudness(outro_audio, sermon_audio)

    # Assemble final audio
    final_audio = assemble_final_audio(intro_audio, sermon_audio, outro_audio, sample_rate)

    # Write final audio to disk
    out_path = sermon_path.replace(".wav", "_final.wav")
    final_audio.export(out_path, format="wav")

    append_file_log(log_buffer, f"intro_outro: wrote final audio to {out_path}")

    # Update state
    state["working_path"] = out_path
    state.setdefault("final_paths", {})["intro_outro"] = out_path

    state.setdefault("actions", []).append({
        "step": "intro_outro",
        "tts_voice": voice,
        "intro_text": intro_text,
        "outro_text": outro_text
    })

    return state


# ---------------------------------------------------------
# DISPATCHER ENTRY POINT
# ---------------------------------------------------------

def run(state, ctx):
    return apply_intro_outro(state, ctx)
