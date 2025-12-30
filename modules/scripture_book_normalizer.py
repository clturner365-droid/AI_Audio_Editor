"""
scripture_book_normalizer.py

Purpose:
    Normalize Bible book names in transcript text using:
        - variant map (variant → (book_num, confidence))
        - canonical map (book_num → canonical_name)
        - guard rails for low-confidence variants
        - scripture-context detection

This module:
    - NEVER modifies the Whisper transcript
    - ONLY normalizes book names for downstream linking
    - Is fully production-ready
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable


# -------------------------------------------------------------------
# File-loading helpers
# -------------------------------------------------------------------

def load_variant_map_from_lines(lines: Iterable[str]) -> Dict[str, Tuple[int, str]]:
    """
    Load variant map from lines of:
        variant|book_number|confidence
    Returns:
        { "genesis": (1, "H"), "name": (34, "L"), ... }
    """
    mapping: Dict[str, Tuple[int, str]] = {}

    for raw in lines:
        line = raw.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|")
        if len(parts) == 2:
            # Backward compatibility: assume high confidence
            variant, num = parts
            conf = "H"
        elif len(parts) == 3:
            variant, num, conf = parts
        else:
            continue

        variant = variant.strip().lower()
        if not variant:
            continue

        book_num = int(num.strip())
        conf = conf.strip().upper()

        if conf not in ("H", "L", "X"):
            conf = "H"  # safe fallback

        mapping[variant] = (book_num, conf)

    return mapping


def load_canonical_map_from_lines(lines: Iterable[str]) -> Dict[int, str]:
    """
    Load canonical map from lines of:
        canonical_name|book_number
    Returns:
        { 1: "genesis", 2: "exodus", ... }
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


# -------------------------------------------------------------------
# Core normalizer
# -------------------------------------------------------------------

@dataclass
class ScriptureBookNormalizer:
    variant_map: Dict[str, Tuple[int, str]]   # variant → (book_num, confidence)
    canonical_map: Dict[int, str]             # book_num → canonical_name

    def __post_init__(self) -> None:
        # Normalize keys
        self.variant_map = {
            k.lower(): (int(v[0]), v[1].upper())
            for k, v in self.variant_map.items()
        }
        self.canonical_map = {
            int(k): v.lower()
            for k, v in self.canonical_map.items()
        }

        # Build regex (longest first)
        escaped = [re.escape(v) for v in self.variant_map]
        escaped.sort(key=len, reverse=True)
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        self.variant_regex = re.compile(pattern, flags=re.IGNORECASE)

        # Precompile context patterns
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
        Replace book-name variants with canonical names,
        using confidence levels and guard rails.
        """

        def replace_variant(match: re.Match) -> str:
            found_original = match.group(1)
            found = found_original.lower()

            if found not in self.variant_map:
                return found_original

            book_num, conf = self.variant_map[found]
            canonical = self.canonical_map[book_num]

            # Disabled variant
            if conf == "X":
                return found_original

            # High confidence → always replace
            if conf == "H":
                return canonical

            # Low confidence → require scripture context
            if conf == "L":
                if self._has_scripture_context(text, match.start(), match.end()):
                    return canonical
                else:
                    return found_original

            # Fallback
            return canonical

        return self.variant_regex.sub(replace_variant, text)

    # ------------------------------------------------------------------
    # Scripture context detection
    # ------------------------------------------------------------------

    def _has_scripture_context(self, text: str, start: int, end: int) -> bool:
        """
        Determine whether a low-confidence variant is being used
        in a scripture-like context.
        """

        window_radius = 40
        left = max(0, start - window_radius)
        right = min(len(text), end + window_radius)
        context = text[left:right].lower()

        # 1. Colon verse pattern (e.g., "1:3")
        if self._colon_pattern.search(context):
            return True

        # 2. Numbers + context words
        if self._number_pattern.search(context):
            if any(w in context for w in self._context_words):
                return True

        # 3. "chapter three" / "verse two"
        if self._chapter_word_pattern.search(context):
            return True
        if self._verse_word_pattern.search(context):
            return True

        return False
