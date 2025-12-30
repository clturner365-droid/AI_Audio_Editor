"""
scripture_book_normalizer.py

Step:
    Pre-link normalization of Bible book names.

Purpose:
    - Detect book-name variants in raw transcript text
    - Normalize variants to canonical book names
    - Use contextual clues (numbers, 'chapter', 'verse', etc.) to
      safely disambiguate low-confidence variants (e.g., 'name' → Nahum)
    - Output a clean, canonicalized transcript string

Inputs:
    - Variant file lines:  "<variant>|<book_number>"
    - Canonical file lines:"<canonical_book_name>|<book_number>"

Guarantees:
    - Never inserts chapter/verse numbers
    - Never performs linking
    - Only changes book-name tokens that can be safely normalized
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Set, Iterable


# -------------------------------------------------------------------
# Low-confidence variants that MUST have scripture-like context
# -------------------------------------------------------------------

LOW_CONFIDENCE_VARIANTS: Set[str] = {
    # You can extend this list as needed, but this module
    # treats all of these as risky everyday English words.
    "name",      # Nahum (dangerous without context)
    "job",       # Job
    "jobs",
    "mark",      # Mark
    "acts",      # Acts
    "act",
    "song",      # Song of Songs / Song of Solomon
    "songs",
    "numbers",   # Numbers
    "judge",     # Judges
    "judges",
    "james",     # James
}


# -------------------------------------------------------------------
# Helper functions to build maps from your metadata files
# -------------------------------------------------------------------

def load_variant_map_from_lines(lines: Iterable[str]) -> Dict[str, int]:
    """
    Given lines like:
        'genesis|1'
        'book of genesis|1'
    return: { 'genesis': 1, 'book of genesis': 1, ... }
    """
    mapping: Dict[str, int] = {}
    for raw in lines:
        line = raw.strip()
        if not line or "|" not in line:
            continue
        variant, num = line.split("|", 1)
        variant = variant.strip().lower()
        if not variant:
            continue
        mapping[variant] = int(num.strip())
    return mapping


def load_canonical_map_from_lines(lines: Iterable[str]) -> Dict[int, str]:
    """
    Given lines like:
        'genesis|1'
        'exodus|2'
    return: { 1: 'genesis', 2: 'exodus', ... }
    """
    mapping: Dict[int, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or "|" not in line:
            continue
        name, num = line.split("|", 1)
        name = name.strip().lower()
        if not name:
            continue
        mapping[int(num.strip())] = name
    return mapping


def load_variant_map_from_file(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return load_variant_map_from_lines(f.readlines())


def load_canonical_map_from_file(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        return load_canonical_map_from_lines(f.readlines())


# -------------------------------------------------------------------
# Core normalizer
# -------------------------------------------------------------------

@dataclass
class ScriptureBookNormalizer:
    """
    Production-ready normalizer for Bible book names.

    Usage:
        variant_map   = load_variant_map_from_file("variants.txt")
        canonical_map = load_canonical_map_from_file("canonical.txt")
        normalizer    = ScriptureBookNormalizer(variant_map, canonical_map)

        normalized = normalizer.normalize_text(raw_transcript)
    """

    variant_map: Dict[str, int]      # e.g. {"genesis": 1, "book genesis": 1, ...}
    canonical_map: Dict[int, str]    # e.g. {1: "genesis", 2: "exodus", ...}

    def __post_init__(self) -> None:
        # Normalize keys/values to lowercase for consistent matching
        self.variant_map = {k.lower(): int(v) for k, v in self.variant_map.items()}
        self.canonical_map = {int(k): v.lower() for k, v in self.canonical_map.items()}

        # Build high/low confidence sets based on the variant keys
        self.low_confidence: Set[str] = {
            v for v in self.variant_map.keys() if v in LOW_CONFIDENCE_VARIANTS
        }
        self.high_confidence: Set[str] = {
            v for v in self.variant_map.keys() if v not in self.low_confidence
        }

        # Precompile regex to match ANY variant as a whole token.
        # Longer variants first to avoid partial matches.
        escaped = [re.escape(v) for v in self.variant_map.keys()]
        escaped.sort(key=len, reverse=True)
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        self.variant_regex = re.compile(pattern, flags=re.IGNORECASE)

        # Precompile scripture-context helpers
        self._number_pattern = re.compile(r"\b[0-9]{1,3}\b")
        self._colon_pattern = re.compile(r"[0-9]{1,3}\s*:\s*[0-9]{1,3}")
        self._chapter_word_pattern = re.compile(
            r"\bchapter\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
            flags=re.IGNORECASE,
        )
        self._verse_word_pattern = re.compile(
            r"\bverse\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
            flags=re.IGNORECASE,
        )
        self._context_words = (
            "chapter",
            "chap",
            "verse",
            "verses",
            "psalm",
            "psalms",
            "book",
            "books",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Replace all variant book names in `text` with canonical names,
        using guard rails for low-confidence variants.

        Returns a new string. Does not modify the input in place.
        """

        def replace_variant(match: re.Match) -> str:
            found_original = match.group(1)
            found = found_original.lower()

            book_num = self.variant_map.get(found)
            if not book_num:
                # Safety fallback: return original token untouched
                return found_original

            canonical = self.canonical_map[book_num]

            # High-confidence variants: always replace
            if found in self.high_confidence:
                return canonical

            # Low-confidence variants: only replace when context is scripture-like
            if found in self.low_confidence:
                if self._has_scripture_context(text, match.start(), match.end()):
                    return canonical
                else:
                    # Leave as-is to avoid corrupting normal prose
                    return found_original

            # Fallback: treat as high-confidence
            return canonical

        return self.variant_regex.sub(replace_variant, text)

    # ------------------------------------------------------------------
    # Scripture context detection (guard rail core)
    # ------------------------------------------------------------------

    def _has_scripture_context(self, text: str, start: int, end: int) -> bool:
        """
        Decide whether a low-confidence variant at [start:end] is being
        used in a scripture-like way.

        We inspect a small window around the token and look for:
            - Nearby chapter/verse colon pattern (e.g., '1:3')
            - Numbers plus typical scripture context words
            - 'chapter N' or 'verse N' phrasing

        This is deliberately conservative.
        """

        window_radius = 40  # characters before/after to inspect
        left_index = max(0, start - window_radius)
        right_index = min(len(text), end + window_radius)
        context = text[left_index:right_index].lower()

        # 1) Colon verse pattern like '1:3' nearby
        if self._colon_pattern.search(context):
            return True

        # 2) Digit-based numbers present?
        if self._number_pattern.search(context):
            if any(word in context for word in self._context_words):
                return True

        # 3) 'chapter three' / 'verse two' style patterns
        if self._chapter_word_pattern.search(context):
            return True
        if self._verse_word_pattern.search(context):
            return True

        return False
