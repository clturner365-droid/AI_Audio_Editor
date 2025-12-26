# registry.py
# Handles loading, updating, and saving the SpeakerRegistry.json

import os
import json
from datetime import datetime

REGISTRY_PATH = os.path.join("PipelineB", "SpeakerRegistry.json")


# ---------------------------------------------------------
# LOADING & SAVING
# ---------------------------------------------------------

def load_registry():
    """
    Loads the SpeakerRegistry.json file.
    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(REGISTRY_PATH):
        return {}

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        # Corrupted registry → start fresh
        return {}


def save_registry(registry):
    """
    Writes the updated registry back to disk.
    """
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------
# SPEAKER LOOKUP
# ---------------------------------------------------------

def find_speaker_by_canonical(registry, name):
    """
    Returns the speaker entry if the canonical name matches.
    """
    return registry.get(name)


def find_speaker_by_alias(registry, name):
    """
    Searches all speakers for a matching alias.
    Returns (canonical_name, speaker_entry) or (None, None).
    """
    for canonical, entry in registry.items():
        aliases = entry.get("aliases", [])
        if name.lower() in [a.lower() for a in aliases]:
            return canonical, entry
    return None, None


# ---------------------------------------------------------
# ADDING NEW SPEAKERS
# ---------------------------------------------------------

def add_new_speaker(registry, canonical_name):
    """
    Creates a new speaker entry with default fields.
    """
    registry[canonical_name] = {
        "aliases": [],
        "fingerprint": None,
        "fingerprint_strength": 0,
        "intro_count": 0,
        "outro_count": 0,
        "last_intro_used": -1,
        "last_outro_used": -1,
        "created": timestamp(),
        "updated": timestamp(),
        "notes": ""
    }
    return registry[canonical_name]


# ---------------------------------------------------------
# ALIAS MANAGEMENT
# ---------------------------------------------------------

def add_alias(registry, canonical_name, alias):
    """
    Adds an alias to a speaker if it doesn't already exist.
    """
    entry = registry.get(canonical_name)
    if entry is None:
        return

    aliases = entry.get("aliases", [])
    if alias.lower() not in [a.lower() for a in aliases]:
        aliases.append(alias)
        entry["aliases"] = aliases
        entry["updated"] = timestamp()


# ---------------------------------------------------------
# FINGERPRINT UPDATES
# ---------------------------------------------------------

def update_fingerprint(registry, canonical_name, fingerprint_vector, strength):
    """
    Updates the speaker's fingerprint and strength.
    Only used for single-speaker files.
    """
    entry = registry.get(canonical_name)
    if entry is None:
        return

    entry["fingerprint"] = fingerprint_vector
    entry["fingerprint_strength"] = strength
    entry["updated"] = timestamp()


# ---------------------------------------------------------
# INTRO/OUTRO ROTATION
# ---------------------------------------------------------

def get_next_intro_index(entry):
    """
    Returns the next intro index for rotation.
    """
    count = entry.get("intro_count", 0)
    last = entry.get("last_intro_used", -1)

    if count == 0:
        return 0  # no intros yet

    return (last + 1) % count


def get_next_outro_index(entry):
    """
    Returns the next outro index for rotation.
    """
    count = entry.get("outro_count", 0)
    last = entry.get("last_outro_used", -1)

    if count == 0:
        return 0

    return (last + 1) % count


def update_intro_rotation(entry, index):
    entry["last_intro_used"] = index
    entry["updated"] = timestamp()


def update_outro_rotation(entry, index):
    entry["last_outro_used"] = index
    entry["updated"] = timestamp()


# ---------------------------------------------------------
# UTILITY
# ---------------------------------------------------------

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
