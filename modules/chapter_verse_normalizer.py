"""
chapter_verse_normalizer.py

Purpose:
    Normalize spoken or loosely formatted chapter/verse references into
    canonical <chapter>:<verse> formats, but ONLY when anchored to a
    canonical Bible book name within a bidirectional window.

Features:
    - Converts number words (one, two, three…) to integers
    - Detects chapter/verse patterns safely
    - Prevents partial-word matches
    - Requires book anchor within ±80 chars
    - Handles:
        * chapter X verse Y
        * X:Y
        * X Y
        * X:Yf / X:Yff
        * X:Y-YY
        * verses X and following → Xff
        * and the next verse → f
        * and the next few verses → ff
    - Pipeline-integrated logging
    - Debug toggle for deep tracing
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Dict


# -------------------------------------------------------------------
# Number word mapping
# -------------------------------------------------------------------

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20
}

def word_to_int(token: str) -> Optional[int]:
    token = token.lower().strip()
    return NUMBER_WORDS.get(token)


# -------------------------------------------------------------------
# Core normalizer
# -------------------------------------------------------------------

@dataclass
class ChapterVerseNormalizer:
    canonical_map: Dict[int, str]  # book_num → canonical_name
    logger: Optional[object] = None
    step_id: str = "normalize.chapter_verse"
    debug: bool = False

    def __post_init__(self):
        # Build book-name detection regex
        self._book_names = list(self.canonical_map.values())
        self._book_pattern = re.compile(
            r"\b(" + "|".join(re.escape(b) for b in self._book_names) + r")\b",
            flags=re.IGNORECASE
        )

        # Patterns for chapter/verse detection
        self._chapter_word = re.compile(
            r"\bchapter\s+(?P<chap>(\d+|one|two|three|four|five|six|seven|eight|nine|ten))\b",
            flags=re.IGNORECASE
        )

        self._verse_word = re.compile(
            r"\bverse\s+(?P<verse>(\d+|one|two|three|four|five|six|seven|eight|nine|ten))\b",
            flags=re.IGNORECASE
        )

        # Compact numeric patterns
        self._compact = re.compile(
            r"\b(?P<chap>\d{1,3})\s*[: ]\s*(?P<verse>\d{1,3})(?P<suffix>f{1,2})?\b",
            flags=re.IGNORECASE
        )

        # Range patterns
        self._range = re.compile(
            r"\b(?P<chap>\d{1,3})\s*[: ]\s*(?P<v1>\d{1,3})\s*-\s*(?P<v2>\d{1,3})\b",
            flags=re.IGNORECASE
        )

        # Following patterns
        self._following = re.compile(
            r"\bverses?\s+(?P<verse>\d{1,3})\s+(and\s+the\s+following|and\s+following)\b",
            flags=re.IGNORECASE
        )

        # Next-verse patterns
        self._next_verse = re.compile(
            r"\b(and\s+the\s+next\s+verse|and\s+next\s+verse)\b",
            flags=re.IGNORECASE
        )

        self._next_few_verses = re.compile(
            r"\b(and\s+the\s+next\s+few\s+verses|and\s+next\s+few\s+verses)\b",
            flags=re.IGNORECASE
        )

    # ------------------------------------------------------------------
    # Book anchor detection
    # ------------------------------------------------------------------

    def _has_book_anchor(self, text: str, pos: int, window: int = 80) -> bool:
        left = max(0, pos - window)
        right = min(len(text), pos + window)
        context = text[left:right].lower()
        return bool(self._book_pattern.search(context))

    # ------------------------------------------------------------------
    # Chapter/verse extraction helpers
    # ------------------------------------------------------------------

    def _find_last_chapter_and_verse(self, text: str, pos: int):
        window = text[max(0, pos - 120):pos].lower()

        # Find last "chapter X"
        chap_match = re.findall(
            r"chapter\s+(?P<chap>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)",
            window,
            flags=re.IGNORECASE
        )
        chap = None
        if chap_match:
            raw = chap_match[-1]
            chap = int(raw) if raw.isdigit() else word_to_int(raw)

        # Find last verse number
        verse_match = re.findall(r"\b(\d{1,3})\b", window)
        verse = int(verse_match[-1]) if verse_match else None

        return chap, verse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Normalize chapter/verse expressions in the text,
        but ONLY when anchored to a book name.
        """

        # Order matters: handle ranges first
        text = self._range.sub(self._replace_range, text)

        # Following patterns
        text = self._following.sub(self._replace_following, text)

        # Next-verse patterns
        text = self._next_few_verses.sub(self._replace_next_few_verses, text)
        text = self._next_verse.sub(self._replace_next_verse, text)

        # Compact patterns
        text = self._compact.sub(self._replace_compact, text)

        # Word patterns
        text = self._chapter_word.sub(self._replace_chapter_word, text)
        text = self._verse_word.sub(self._replace_verse_word, text)

        return text

    # ------------------------------------------------------------------
    # Replacement handlers
    # ------------------------------------------------------------------

    def _replace_range(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        chap = int(match.group("chap"))
        v1 = int(match.group("v1"))
        v2 = int(match.group("v2"))
        out = f"{chap}:{v1}-{v2}"

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "range_normalized",
                "input": match.group(0),
                "output": out
            })

        return out

    def _replace_compact(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        chap = int(match.group("chap"))
        verse = int(match.group("verse"))
        suffix = match.group("suffix") or ""
        out = f"{chap}:{verse}{suffix}"

        if self.debug and self.logger:
            self.logger.debug({
                "step": self.step_id,
                "event": "compact_normalized",
                "input": match.group(0),
                "output": out
            })

        return out

    def _replace_chapter_word(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        chap_raw = match.group("chap")
        chap = int(chap_raw) if chap_raw.isdigit() else word_to_int(chap_raw)
        out = f"{chap}:"

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "chapter_word_normalized",
                "input": match.group(0),
                "output": out
            })

        return out

    def _replace_verse_word(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        verse_raw = match.group("verse")
        verse = int(verse_raw) if verse_raw.isdigit() else word_to_int(verse_raw)
        out = f"{verse}"

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "verse_word_normalized",
                "input": match.group(0),
                "output": out
            })

        return out

    def _replace_following(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        verse = int(match.group("verse"))
        chap, _ = self._find_last_chapter_and_verse(match.string, match.start())
        if chap is None:
            return match.group(0)

        out = f"{chap}:{verse}ff"

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "following_normalized",
                "input": match.group(0),
                "output": out
            })

        return out

    def _replace_next_verse(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        chap, verse = self._find_last_chapter_and_verse(match.string, match.start())
        if chap is None or verse is None:
            return match.group(0)

        out = f"{chap}:{verse}f"

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "next_verse_normalized",
                "input": match.group(0),
                "output": out
            })

        return out

    def _replace_next_few_verses(self, match: re.Match) -> str:
        if not self._has_book_anchor(match.string, match.start()):
            return match.group(0)

        chap, verse = self._find_last_chapter_and_verse(match.string, match.start())
        if chap is None or verse is None:
            return match.group(0)

        out = f"{chap}:{verse}ff"

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "next_few_verses_normalized",
                "input": match.group(0),
                "output": out
            })

        return out
