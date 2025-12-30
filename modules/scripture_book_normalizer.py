"""
scripture_book_normalizer.py
Step: Pre-link normalization
Purpose:
    - Detect book names in raw transcript text
    - Normalize variants to canonical book names
    - Use contextual clues (numbers before/after) to disambiguate
    - Output a clean, canonicalized transcript
"""

import re
from typing import Dict, List, Tuple


class ScriptureBookNormalizer:
    def __init__(
        self,
        variant_map: Dict[str, int],
        canonical_map: Dict[int, str],
    ):
        """
        variant_map: { "gen": 1, "gennises": 1, ... }
        canonical_map: { 1: "genesis", 2: "exodus", ... }
        """
        self.variant_map = variant_map
        self.canonical_map = canonical_map

        # Precompile regex for speed
        self.variant_regex = self._build_variant_regex()

    def _build_variant_regex(self) -> re.Pattern:
        """
        Build a single giant regex that matches ANY variant.
        """
        escaped = [re.escape(v) for v in self.variant_map.keys()]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, flags=re.IGNORECASE)

    def normalize(self, text: str) -> str:
        """
        Main entry point.
        Replace all variant book names with canonical names.
        """

        def replace_variant(match: re.Match) -> str:
            found = match.group(1).lower()
            book_num = self.variant_map.get(found)
            if not book_num:
                return found  # shouldn't happen, but safe fallback
            canonical = self.canonical_map[book_num]
            return canonical

        # First pass: replace variants with canonical names
        text = self.variant_regex.sub(replace_variant, text)

        # Second pass: apply contextual disambiguation
        text = self._apply_contextual_rules(text)

        return text

    def _apply_contextual_rules(self, text: str) -> str:
        """
        Use textual clues to refine detection:
            - Numbers before book names (e.g., "1 john")
            - Numbers after book names (e.g., "john 3")
            - Spoken forms like "chapter three"
        """

        # Example rule: normalize "1 john" → "first john"
        text = re.sub(
            r"\b1\s+john\b",
            "first john",
            text,
            flags=re.IGNORECASE,
        )

        # Example rule: normalize "2 kings" → "second kings"
        text = re.sub(
            r"\b2\s+kings\b",
            "second kings",
            text,
            flags=re.IGNORECASE,
        )

        # Add more rules as needed
        return text
