"""
modules/contextual_rules.py

Purpose:
    Apply contextual cleanup rules to transcript text BEFORE
    scripture book normalization and chapter/verse normalization.

Dispatcher-ready:
    - No module calls another module
    - Uses structured logging
    - Step-ID aware
    - Debug toggle
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextualRules:
    logger: Optional[object] = None
    step_id: str = "scripture.contextual_rules"
    debug: bool = False

    # Spoken ordinals → numeric
    ORDINAL_MAP = {
        "first": "1",
        "second": "2",
        "third": "3",
        "fourth": "4",
        "fifth": "5",
    }

    # Spoken cardinals → numeric
    CARDINAL_MAP = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
    }

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def apply(self, text: str) -> str:
        """Apply all contextual cleanup rules to text."""
        if self.debug and self.logger:
            self.logger.debug({
                "step": self.step_id,
                "event": "start_contextual_rules",
                "input_preview": text[:120]
            })

        original = text

        text = self._normalize_spoken_ordinals(text)
        text = self._normalize_spoken_cardinals(text)
        text = self._fix_number_before_book(text)
        text = self._fix_number_after_book(text)
        text = self._fix_chapter_patterns(text)
        text = self._fix_verse_patterns(text)
        text = self._fix_misordered_patterns(text)

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "contextual_rules_applied",
                "changed": (text != original)
            })

        if self.debug and self.logger:
            self.logger.debug({
                "step": self.step_id,
                "event": "end_contextual_rules",
                "output_preview": text[:120]
            })

        return text

    # ------------------------------------------------------------
    # 1. Normalize spoken ordinals (first john → 1 john)
    # ------------------------------------------------------------
    def _normalize_spoken_ordinals(self, text: str) -> str:
        for word, num in self.ORDINAL_MAP.items():
            text = re.sub(
                rf"\b{word}\s+(john|peter|corinthians|samuel|kings|chronicles)\b",
                f"{num} \\1",
                text,
                flags=re.IGNORECASE,
            )
        return text

    # ------------------------------------------------------------
    # 2. Normalize spoken cardinals (one john → 1 john)
    # ------------------------------------------------------------
    def _normalize_spoken_cardinals(self, text: str) -> str:
        for word, num in self.CARDINAL_MAP.items():
            text = re.sub(
                rf"\b{word}\s+(john|peter|corinthians|samuel|kings|chronicles)\b",
                f"{num} \\1",
                text,
                flags=re.IGNORECASE,
            )
        return text

    # ------------------------------------------------------------
    # 3. Fix patterns like "1 john" or "2 kings"
    # ------------------------------------------------------------
    def _fix_number_before_book(self, text: str) -> str:
        return re.sub(
            r"\b([1-3])\s+(john|peter|corinthians|samuel|kings|chronicles)\b",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------
    # 4. Fix patterns like "john 3" (book + chapter)
    # ------------------------------------------------------------
    def _fix_number_after_book(self, text: str) -> str:
        return re.sub(
            r"\b(john|romans|psalms|proverbs|isaiah|jeremiah)\s+([0-9]+)\b",
            r"\1 \2",
            text,
            flags=re.IGNORECASE,
        )

    # ------------------------------------------------------------
    # 5. Normalize "chapter three" → "chapter 3"
    # ------------------------------------------------------------
    def _fix_chapter_patterns(self, text: str) -> str:
        for word, num in self.CARDINAL_MAP.items():
            text = re.sub(
                rf"\bchapter\s+{word}\b",
                f"chapter {num}",
                text,
                flags=re.IGNORECASE,
            )
        return text

    # ------------------------------------------------------------
    # 6. Normalize "verse twenty one" → "verse 21"
    # ------------------------------------------------------------
    def _fix_verse_patterns(self, text: str) -> str:
        for word, num in self.CARDINAL_MAP.items():
            text = re.sub(
                rf"\bverse\s+{word}\b",
                f"verse {num}",
                text,
                flags=re.IGNORECASE,
            )
        return text

    # ------------------------------------------------------------
    # 7. Fix misordered patterns ("chapter 3 of john")
    # ------------------------------------------------------------
    def _fix_misordered_patterns(self, text: str) -> str:
        return re.sub(
            r"\bchapter\s+([0-9]+)\s+of\s+(john|romans|psalms|proverbs)\b",
            r"\2 \1",
            text,
            flags=re.IGNORECASE,
        )


# ----------------------------------------------------------------------
# Dispatcher wrapper
# ----------------------------------------------------------------------

def run(state, ctx):
    """
    Dispatcher entry point for contextual scripture cleanup.
    """

    logger = ctx.get("logger")
    debug = bool(ctx.get("debug", False))
    step_id = ctx.get("step_id", "scripture.contextual_rules")

    if logger:
        logger.info({
            "step": step_id,
            "event": "contextual_rules_start"
        })

    rules = ContextualRules(
        logger=logger,
        step_id=step_id,
        debug=debug
    )

    text = state.get("transcript", "")
    cleaned = rules.apply(text)
    state["transcript"] = cleaned

    if logger:
        logger.info({
            "step": step_id,
            "event": "contextual_rules_complete"
        })

    return state
