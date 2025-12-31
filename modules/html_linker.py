"""
html_linker.py

Purpose:
    Convert normalized scripture references (book chapter:verse)
    into HTML links, validate references using verse-count metadata,
    and mark invalid references in grey text.

Features:
    - Detects normalized references:
        * book chapter:verse
        * book chapter:verse-verse
        * book chapter:versef / verseff
    - Validates references using verse_map
    - Wraps valid references in:
        <span class="scripture-ref"><a href="...">Book 3:10</a></span>
    - Marks invalid references as:
        <span class="invalid-ref">Book 3:99 (invalid)</span>
    - Pipeline-integrated logging
    - Debug toggle for deep tracing
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# -------------------------------------------------------------------
# Core HTML Linker
# -------------------------------------------------------------------

@dataclass
class ScriptureHTMLLinker:
    verse_map: Dict[int, Dict[int, int]]      # book_num → {chapter: verse_count}
    canonical_map: Dict[int, str]             # book_num → canonical_name
    logger: Optional[object] = None
    step_id: str = "link.scripture"
    debug: bool = False

    def __post_init__(self):
        # Build book-name detection regex
        self._book_names = list(self.canonical_map.values())
        self._book_pattern = re.compile(
            r"\b(" + "|".join(re.escape(b) for b in self._book_names) + r")\b",
            flags=re.IGNORECASE
        )

        # Detect normalized references:
        # book chapter:verse
        # book chapter:verse-verse
        # book chapter:versef / verseff
        self._ref_pattern = re.compile(
            r"\b(?P<book>" + "|".join(re.escape(b) for b in self._book_names) + r")\s+"
            r"(?P<chap>\d{1,3})\s*:\s*"
            r"(?P<verse>\d{1,3})"
            r"(?P<suffix>f{1,2})?"
            r"(?:-(?P<endverse>\d{1,3}))?"
            r"\b",
            flags=re.IGNORECASE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(self, text: str) -> str:
        """
        Convert normalized scripture references into HTML links,
        validating each reference using verse_map.
        """
        return self._ref_pattern.sub(self._replace_reference, text)

    # ------------------------------------------------------------------
    # Replacement handler
    # ------------------------------------------------------------------

    def _replace_reference(self, match: re.Match) -> str:
        book_raw = match.group("book").lower()
        chap = int(match.group("chap"))
        verse = int(match.group("verse"))
        suffix = match.group("suffix") or ""
        endverse_raw = match.group("endverse")

        # Resolve book number
        book_num = self._resolve_book_number(book_raw)
        canonical = self.canonical_map[book_num]

        # Validate reference
        is_valid, reason = self._validate_reference(
            book_num, chap, verse, suffix, endverse_raw
        )

        # Build display text
        if endverse_raw:
            display = f"{canonical} {chap}:{verse}-{endverse_raw}"
        else:
            display = f"{canonical} {chap}:{verse}{suffix}"

        if is_valid:
            link = self._build_link(canonical, chap, verse)
            html = f'<span class="scripture-ref"><a href="{link}">{display}</a></span>'

            if self.logger:
                self.logger.info({
                    "step": self.step_id,
                    "event": "valid_reference_linked",
                    "reference": display,
                    "link": link
                })

            return html

        # Invalid reference
        html = (
            f'<span class="invalid-ref">{display} (invalid)</span>'
        )

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "invalid_reference_detected",
                "reference": display,
                "reason": reason
            })

        return html

    # ------------------------------------------------------------------
    # Book number resolution
    # ------------------------------------------------------------------

    def _resolve_book_number(self, book: str) -> int:
        for num, name in self.canonical_map.items():
            if name == book:
                return num
        raise ValueError(f"Unknown book: {book}")

    # ------------------------------------------------------------------
    # Reference validation
    # ------------------------------------------------------------------

    def _validate_reference(
        self,
        book_num: int,
        chap: int,
        verse: int,
        suffix: str,
        endverse_raw: Optional[str]
    ) -> Tuple[bool, str]:

        # Check chapter exists
        if chap not in self.verse_map.get(book_num, {}):
            return False, "chapter does not exist"

        max_verse = self.verse_map[book_num][chap]

        # Check verse exists
        if verse < 1 or verse > max_verse:
            return False, "verse out of range"

        # Range validation
        if endverse_raw:
            endverse = int(endverse_raw)
            if endverse < verse or endverse > max_verse:
                return False, "invalid verse range"

        # f / ff validation
        if suffix == "f":
            if verse == max_verse:
                return False, "f extends beyond chapter"
        elif suffix == "ff":
            if verse >= max_verse:
                return False, "ff extends beyond chapter"

        return True, "valid"

    # ------------------------------------------------------------------
    # Link builder
    # ------------------------------------------------------------------

    def _build_link(self, canonical: str, chap: int, verse: int) -> str:
        # Example:
        # https://eecc.org/kjv/matthew/3.html#v10
        return f"https://eecc.org/kjv/{canonical}/{chap}.html#v{verse}"
