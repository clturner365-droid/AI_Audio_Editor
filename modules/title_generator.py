#!/usr/bin/env python3
"""
modules/title_generator.py

Generates a short, natural sermon title from the transcript using a
CPU‑only summarization model. Loaded once at startup in main.py.

- Uses a distilled summarization model (0 VRAM)
- Produces 3–8 word titles
- Skips generation if metadata already contains a title
- Safe, broadcast‑friendly phrasing
"""

from transformers import pipeline
from modules.logging_system import append_file_log


# ---------------------------------------------------------
# MODEL LOADING (CPU‑ONLY)
# ---------------------------------------------------------

def load_title_model(log_buffer=None):
    """
    Loads a small summarization model on CPU.
    This model uses ~1–1.5 GB RAM and 0 VRAM.
    """
    if log_buffer:
        append_file_log(log_buffer, "Loading CPU title‑generation model...")

    # Distilled summarization model (fast, small, CPU‑friendly)
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # Force CPU
    )

    if log_buffer:
        append_file_log(log_buffer, "Title‑generation model loaded (CPU‑only).")

    return summarizer


# ---------------------------------------------------------
# TITLE GENERATION
# ---------------------------------------------------------

def generate_title_from_transcript(transcript: str, model, log_buffer=None) -> str:
    """
    Generates a short, natural sermon title from the transcript.
    Returns a 3–8 word title.

    If transcript is empty, returns a safe fallback.
    """

    if not transcript or len(transcript.strip()) < 40:
        if log_buffer:
            append_file_log(log_buffer, "Transcript too short for title generation; using fallback.")
        return "A Message of Hope"

    if log_buffer:
        append_file_log(log_buffer, "Generating sermon title from transcript...")

    # Summarize the transcript into a short sentence
    summary = model(
        transcript,
        max_length=20,
        min_length=5,
        do_sample=False
    )[0]["summary_text"]

    # Convert summary into a short title
    title = _condense_to_title(summary)

    if log_buffer:
        append_file_log(log_buffer, f"Generated title: {title}")

    return title


# ---------------------------------------------------------
# TITLE CLEANUP / CONDENSING
# ---------------------------------------------------------

def _condense_to_title(summary: str) -> str:
    """
    Converts a short summary into a clean, 3–8 word title.
    Ensures capitalization and removes trailing punctuation.
    """

    # Remove punctuation at the end
    summary = summary.strip().rstrip(".!?")

    # Split into words
    words = summary.split()

    # Keep 3–8 words
    words = words[:8]
    if len(words) < 3:
        return "A Message of Hope"

    # Capitalize each word
    title = " ".join(w.capitalize() for w in words)

    return title
