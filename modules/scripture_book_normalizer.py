"""
scripture_book_normalizer.py

Pipeline-integrated version with debug toggle.

Features:
    - Variant normalization with H/L/X confidence
    - Scripture-context detection for L variants
    - Explicit prevention of partial-word matches
    - Pipeline-integrated logging (step_id + shared logger)
    - Debug toggle for deep tracing
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Optional


# -------------------------------------------------------------------
# File-loading helpers
# -------------------------------------------------------------------

def load_variant_map_from_lines(lines: Iterable[str]) -> Dict[str, Tuple[int, str]]:
    mapping: Dict[str, Tuple[int, str]] = {}

    for raw in lines:
        line = raw.strip()
        if not line or "|" not in line:
            continue

        parts = line.split("|")
        if len(parts) == 2:
            variant, num = parts
            conf = "H"
        elif len(parts) == 3:
            variant, num, conf = parts
        else:
            continue

        variant = variant.strip().lower()
        if not variant:
            continue

        mapping[variant] = (int(num.strip()), conf.strip().upper())

    return mapping


def load_canonical_map_from_lines(lines: Iterable[str]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}

    for raw in lines:
        line = raw.strip()
        if not line or "|" not in line:
            continue

        name, num = line.split("|", 1)
        mapping[int(num.strip())] = name.strip().lower()

    return mapping


# -------------------------------------------------------------------
# Core normalizer
# -------------------------------------------------------------------

@dataclass
class ScriptureBookNormalizer:
    variant_map: Dict[str, Tuple[int, str]]
    canonical_map: Dict[int, str]

    # Pipeline logging integration
    logger: Optional[object] = None
    step_id: str = "normalize.scripture"
    debug: bool = False

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

        # Explicitly forbid partial-word matches
        pattern = r"(?<![A-Za-z])(" + "|".join(escaped) + r")(?![A-Za-z])"
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
            "chapter", "chap", "verse", "verses", "psalm", "psalms", "book", "books"
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

            # Debug: log every match
            if self.debug and self.logger:
                self.logger.debug({
                    "step": self.step_id,
                    "event": "variant_matched",
                    "variant": found,
                    "confidence": conf,
                    "book": book_num,
                })

            # Disabled variant
            if conf == "X":
                if self.logger:
                    self.logger.info({
                        "step": self.step_id,
                        "event": "variant_skipped_X",
                        "variant": found,
                    })
                return found_original

            # High confidence → always replace
            if conf == "H":
                if self.debug and self.logger:
                    self.logger.debug({
                        "step": self.step_id,
                        "event": "variant_replaced_H",
                        "variant": found,
                        "canonical": canonical,
                    })
                return canonical

            # Low confidence → require scripture context
            if conf == "L":
                has_context = self._has_scripture_context(text, match.start(), match.end())

                if has_context:
                    if self.logger:
                        self.logger.info({
                            "step": self.step_id,
                            "event": "variant_replaced_L",
                            "variant": found,
                            "canonical": canonical,
                        })
                    return canonical
                else:
                    if self.logger:
                        self.logger.info({
                            "step": self.step_id,
                            "event": "variant_rejected_L",
                            "variant": found,
                        })
                    return found_original

            return canonical

        return self.variant_regex.sub(replace_variant, text)

    # ------------------------------------------------------------------
    # Scripture context detection
    # ------------------------------------------------------------------

    def _has_scripture_context(self, text: str, start: int, end: int) -> bool:
        window_radius = 40
        left = max(0, start - window_radius)
        right = min(len(text), end + window_radius)
        context = text[left:right].lower()

        if self._colon_pattern.search(context):
            return True

        if self._number_pattern.search(context):
            if any(w in context for w in self._context_words):
                return True

        if self._chapter_word_pattern.search(context):
            return True

        if self._verse_word_pattern.search(context):
            return True

        return False
