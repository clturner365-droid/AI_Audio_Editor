import re

class ContextualRules:
    """
    Applies contextual clues to refine book-name normalization.
    This does NOT perform linking — only book-name cleanup.
    """

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

    def apply(self, text: str) -> str:
        text = self._normalize_spoken_ordinals(text)
        text = self._normalize_spoken_cardinals(text)
        text = self._fix_number_before_book(text)
        text = self._fix_number_after_book(text)
        text = self._fix_chapter_patterns(text)
        text = self._fix_verse_patterns(text)
        text = self._fix_misordered_patterns(text)
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
