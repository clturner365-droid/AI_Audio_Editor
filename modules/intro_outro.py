#!/usr/bin/env python3
"""
modules/intro_outro.py

Generates intro + outro using Coqui TTS (same voice for both),
applies duration‑based rules, rotates templates, shifts transcript timestamps,
and assembles final audio using pydub.
"""

import os
from pydub import AudioSegment

from modules.logging_system import append_file_log
from modules.tts_audio_generator import generate_tts_audio


# ---------------------------------------------------------
# TEMPLATE SETS (FINAL, PRODUCTION‑READY)
# ---------------------------------------------------------

INTRO_TEMPLATES = [
    "Today’s message is titled '{title}', delivered by {speaker}.",
    "You’re listening to {speaker} presenting the lesson '{title}'.",
    "Welcome. {speaker} brings us the message '{title}'.",
    "Our speaker today is {speaker}, sharing the lesson '{title}'.",
    "The message for today is '{title}', brought to us by {speaker}."
]

SHORT_INTRO_TEMPLATES = [
    "This is a message by {speaker}.",
    "You’re listening to a lesson from {speaker}.",
    "Today’s speaker is {speaker}.",
    "Here is a short message from {speaker}.",
    "Now presenting a message by {speaker}."
]

OUTRO_TEMPLATES_SPEAKER_ONLY = [
    "This concludes the message by {speaker}.",
    "Thank you for listening to this lesson from {speaker}.",
    "We appreciate you joining us for this message by {speaker}.",
    "This has been a message from {speaker}. Thank you for listening.",
    "Thank you for spending this time with {speaker}."
]

OUTRO_TEMPLATES_TITLE_AND_SPEAKER = [
    "This concludes the lesson '{title}' by {speaker}.",
    "Thank you for listening to '{title}', presented by {speaker}.",
    "We appreciate you joining us for the message '{title}' from {speaker}.",
    "This has been '{title}', delivered by {speaker}. Thank you for listening.",
    "Thank you for spending this time with {speaker} and the lesson '{title}'."
]


# ---------------------------------------------------------
# TEXT GENERATION
# ---------------------------------------------------------

def build_intro_text(state, duration_minutes, file_index):
    speaker = state.get("speaker_name", "your speaker")
    title = state.get("sermon_title", "today's message")

    if duration_minutes < 10:
        template = SHORT_INTRO_TEMPLATES[file_index % len(SHORT_INTRO_TEMPLATES)]
        return template.format(speaker=speaker)

    template = INTRO_TEMPLATES[file_index % len(INTRO_TEMPLATES)]
    return template.format(speaker=speaker, title=title)


def build_outro_text(state, duration_minutes, file_index):
    speaker = state.get("speaker_name", "your speaker")
    title = state.get("sermon_title", "this message")

    if duration_minutes < 10:
        return None

    if duration_minutes < 20:
        template = OUTRO_TEMPLATES_SPEAKER_ONLY[file_index % len(OUTRO_TEMPLATES_SPEAKER_ONLY)]
        return template.format(speaker=speaker)

    template = OUTRO_TEMPLATES_TITLE_AND_SPEAKER[file_index % len(OUTRO_TEMPLATES_TITLE_AND_SPEAKER)]
    return template.format(speaker=speaker, title=title)


# ---------------------------------------------------------
# FINAL ASSEMBLY
# ---------------------------------------------------------

def add_silence(seconds, sample_rate):
    return AudioSegment.silent(duration=int(seconds * 1000), frame_rate=sample_rate)


def assemble_final_audio(intro, sermon, outro, sample_rate):
    silence = add_silence(10, sample_rate)
    return intro + sermon + outro + silence


# ---------------------------------------------------------
# TRANSCRIPT TIMESTAMP SHIFTING
# ---------------------------------------------------------

def shift_transcript_timestamps(transcript, intro_duration_sec, log_buffer):
    """
    Shifts all transcript timestamps by the intro duration.
    Does NOT modify transcript text.
    """
    if transcript is None:
        append_file_log(log_buffer, "intro_outro: no transcript found; skipping timestamp shift.")
        return None

    if "segments" not in transcript:
        append_file_log(log_buffer, "intro_outro: transcript missing segments; skipping timestamp shift.")
        return transcript

    append_file_log(log_buffer, f"intro_outro: shifting transcript timestamps by {intro_duration_sec:.2f} seconds.")

    for seg in transcript["segments"]:
        seg["start"] += intro_duration_sec
        seg["end"] += intro_duration_sec

    return transcript


# ---------------------------------------------------------
# MAIN PIPELINE FUNCTION
# ---------------------------------------------------------

def apply_intro_outro(state, ctx):
    log_buffer = ctx["log_buffer"]
    append_file_log(log_buffer, "intro_outro: starting")

    voice = state.get("tts_voice")
    if not voice:
        append_file_log(log_buffer, "intro_outro: no tts_voice; skipping")
        return state

    sermon_path = state.get("working_path")
    if not sermon_path or not os.path.exists(sermon_path):
        append_file_log(log_buffer, "intro_outro: missing sermon audio")
        return state

    sermon_audio = AudioSegment.from_wav(sermon_path)
    sample_rate = sermon_audio.frame_rate
    duration_minutes = len(sermon_audio) / 1000 / 60
    file_index = ctx.get("file_index", 0)

    # Build intro/outro text
    intro_text = build_intro_text(state, duration_minutes, file_index)
    outro_text = build_outro_text(state, duration_minutes, file_index)

    # Generate TTS audio
    intro_audio = generate_tts_audio(intro_text, speaker=voice, log_buffer=log_buffer)
    intro_duration_sec = len(intro_audio) / 1000.0

    if outro_text:
        outro_audio = generate_tts_audio(outro_text, speaker=voice, log_buffer=log_buffer)
    else:
        outro_audio = AudioSegment.silent(duration=0)

    # Shift transcript timestamps
    transcript = state.get("transcript")
    shifted_transcript = shift_transcript_timestamps(transcript, intro_duration_sec, log_buffer)
    state["transcript"] = shifted_transcript

    # Assemble final audio
    final_audio = assemble_final_audio(intro_audio, sermon_audio, outro_audio, sample_rate)

    out_path = sermon_path.replace(".wav", "_final.wav")
    final_audio.export(out_path, format="wav")

    append_file_log(log_buffer, f"intro_outro: wrote final audio to {out_path}")

    state["working_path"] = out_path
    return state


def run(state, ctx):
    return apply_intro_outro(state, ctx)
