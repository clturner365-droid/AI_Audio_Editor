# metadata.py
# Handles speaker name extraction, normalization, and multi-speaker detection.

import re
from modules.registry import (
    find_speaker_by_canonical,
    find_speaker_by_alias,
    add_new_speaker,
    add_alias
)


# ---------------------------------------------------------
# METADATA EXTRACTION
# ---------------------------------------------------------

def extract_speaker_names(raw_metadata):
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
# NAME NORMALIZATION
# ---------------------------------------------------------

def normalize_name(name):
    """
    Normalizes a speaker name for comparison.
    Example:
        'Alan E. Highers' → 'alan e highers'
    """
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)  # remove punctuation
    name = re.sub(r"\s+", " ", name)     # collapse whitespace
    return name


# ---------------------------------------------------------
# MULTI-SPEAKER DETECTION
# ---------------------------------------------------------

def is_multi_speaker(name_list):
    """
    Returns True if more than one speaker is listed.
    """
    return len(name_list) > 1


# ---------------------------------------------------------
# SPEAKER RESOLUTION
# ---------------------------------------------------------

def resolve_speakers(registry, raw_names, log_buffer):
    """
    Given a list of raw speaker names, resolve them to canonical names.
    Steps:
        1. Normalize names
        2. Try canonical match
        3. Try alias match
        4. Create new speaker if needed
    Returns:
        - canonical_names: list of canonical speaker names
        - multi: True/False
    """

    canonical_names = []

    for raw in raw_names:
        norm = normalize_name(raw)

        # 1. Try canonical match
        entry = find_speaker_by_canonical(registry, raw)
        if entry:
            canonical_names.append(raw)
            continue

        # 2. Try alias match
        alias_match, alias_entry = find_speaker_by_alias(registry, raw)
        if alias_match:
            canonical_names.append(alias_match)
            continue

        # 3. Try canonical match using normalized name
        entry = find_speaker_by_canonical(registry, norm)
        if entry:
            canonical_names.append(norm)
            continue

        # 4. Try alias match using normalized name
        alias_match, alias_entry = find_speaker_by_alias(registry, norm)
        if alias_match:
            canonical_names.append(alias_match)
            continue

        # 5. No match → create new speaker
        new_entry = add_new_speaker(registry, raw)
        canonical_names.append(raw)

        log_buffer.append(f"[INFO] New speaker added to registry: {raw}")

    return canonical_names, is_multi_speaker(canonical_names)


# ---------------------------------------------------------
# ALIAS HANDLING
# ---------------------------------------------------------

def update_aliases_if_needed(registry, canonical_name, raw_name, log_buffer):
    """
    If raw_name is not the canonical name and not already an alias,
    add it as an alias.
    """

    if normalize_name(raw_name) == normalize_name(canonical_name):
        return  # identical, no alias needed

    entry = registry.get(canonical_name)
    if not entry:
        return

    aliases = entry.get("aliases", [])
    if raw_name.lower() not in [a.lower() for a in aliases]:
        add_alias(registry, canonical_name, raw_name)
        log_buffer.append(f"[INFO] Alias added: '{raw_name}' → '{canonical_name}'")
