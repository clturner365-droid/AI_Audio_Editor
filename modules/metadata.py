"""
metadata.py

Purpose:
    Extract speaker names from metadata, normalize them,
    resolve them against the speaker registry, and update aliases.

This module is dispatcher-ready:
    - No module calls another module except registry helpers
    - Uses structured logging (logger.info / logger.debug)
    - Step-ID aware
    - Debug toggle
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from modules.registry import (
    find_speaker_by_canonical,
    find_speaker_by_alias,
    add_new_speaker,
    add_alias
)


# ---------------------------------------------------------
# NAME NORMALIZATION
# ---------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalizes a speaker name for comparison.
    Example:
        'Alan E. Highers' → 'alan e highers'
    """
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)      # remove punctuation
    name = re.sub(r"\s+", " ", name)         # collapse whitespace
    return name


# ---------------------------------------------------------
# METADATA EXTRACTION
# ---------------------------------------------------------

def extract_speaker_names(raw_metadata: str) -> List[str]:
    """
    Extracts speaker names from raw metadata.
    raw_metadata may come from:
        - filename
        - embedded tags
        - external metadata file
    Returns a list of speaker name strings.
    """

    if not raw_metadata:
        return []

    # Split on common separators
    parts = re.split(r"[,&/;]+", raw_metadata)

    # Clean whitespace and remove empties
    names = [p.strip() for p in parts if p.strip()]

    return names


# ---------------------------------------------------------
# MULTI-SPEAKER DETECTION
# ---------------------------------------------------------

def is_multi_speaker(name_list: List[str]) -> bool:
    """Returns True if more than one speaker is listed."""
    return len(name_list) > 1


# ---------------------------------------------------------
# DISPATCHER-READY RESOLUTION CLASS
# ---------------------------------------------------------

@dataclass
class SpeakerResolver:
    logger: Optional[object] = None
    step_id: str = "speaker.resolve"
    debug: bool = False

    # -----------------------------------------------------
    # Resolve speakers
    # -----------------------------------------------------

    def resolve(self, registry: dict, raw_names: List[str]) -> Tuple[List[str], bool]:
        """
        Resolve raw speaker names to canonical names.

        Steps:
            1. Normalize names
            2. Try canonical match
            3. Try alias match
            4. Create new speaker if needed

        Returns:
            (canonical_names, multi_speaker_flag)
        """

        canonical_names = []

        for raw in raw_names:
            norm = normalize_name(raw)

            # Debug logging
            if self.debug and self.logger:
                self.logger.debug({
                    "step": self.step_id,
                    "event": "normalize_name",
                    "raw": raw,
                    "normalized": norm
                })

            # 1. Try canonical match (raw)
            entry = find_speaker_by_canonical(registry, raw)
            if entry:
                canonical_names.append(entry["canonical"])
                self._log_match("canonical_raw", raw, entry["canonical"])
                continue

            # 2. Try alias match (raw)
            alias_match, alias_entry = find_speaker_by_alias(registry, raw)
            if alias_match:
                canonical_names.append(alias_match)
                self._log_match("alias_raw", raw, alias_match)
                continue

            # 3. Try canonical match (normalized)
            entry = find_speaker_by_canonical(registry, norm)
            if entry:
                canonical_names.append(entry["canonical"])
                self._log_match("canonical_norm", raw, entry["canonical"])
                continue

            # 4. Try alias match (normalized)
            alias_match, alias_entry = find_speaker_by_alias(registry, norm)
            if alias_match:
                canonical_names.append(alias_match)
                self._log_match("alias_norm", raw, alias_match)
                continue

            # 5. No match → create new speaker
            new_entry = add_new_speaker(registry, raw)
            canonical_names.append(new_entry["canonical"])

            if self.logger:
                self.logger.info({
                    "step": self.step_id,
                    "event": "new_speaker_added",
                    "raw_name": raw,
                    "canonical": new_entry["canonical"]
                })

        return canonical_names, is_multi_speaker(canonical_names)

    # -----------------------------------------------------
    # Alias update
    # -----------------------------------------------------

    def update_alias(self, registry: dict, canonical_name: str, raw_name: str):
        """
        Add raw_name as an alias if it is not identical to the canonical name.
        """

        if normalize_name(raw_name) == normalize_name(canonical_name):
            return  # identical, no alias needed

        entry = registry.get(canonical_name)
        if not entry:
            return

        aliases = entry.get("aliases", [])
        if raw_name.lower() not in [a.lower() for a in aliases]:
            add_alias(registry, canonical_name, raw_name)

            if self.logger:
                self.logger.info({
                    "step": self.step_id,
                    "event": "alias_added",
                    "raw_name": raw_name,
                    "canonical": canonical_name
                })

    # -----------------------------------------------------
    # Internal logging helper
    # -----------------------------------------------------

    def _log_match(self, match_type: str, raw: str, canonical: str):
        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": f"match_{match_type}",
                "raw_name": raw,
                "canonical": canonical
            })
