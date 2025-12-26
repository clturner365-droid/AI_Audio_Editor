import random

def generate_dynamic_outro(metadata):
    """
    Generates a spoken outro based on sermon duration and metadata.
    Duration rules:
        - >= 20 min: include title + speaker + website
        - 15–20 min: include title + speaker
        - < 15 min: speaker only
    Returns a string ready for TTS.
    """

    speaker = metadata.get("speaker_name", "the speaker")
    title = metadata.get("lesson_title", "this lesson")
    duration_sec = metadata.get("sermon_duration_sec", 0)
    duration_min = duration_sec / 60

    # --- Category 1: Full outro (>= 20 minutes) ---
    if duration_min >= 20:
        options = [
            f"This concludes the lesson {title} by {speaker}. "
            "To hear this message again or explore more lessons, visit us at WSOJ.NET.",

            f"You’ve been listening to {title}, presented by {speaker}. "
            "For this and other lessons, we invite you to visit WSOJ.NET.",

            f"That was {title} from {speaker}. "
            "You can hear this lesson again anytime at WSOJ.NET.",

            f"This has been {title}, delivered by {speaker}. "
            "To replay this message or find more teachings, head over to WSOJ.NET.",

            f"{title} by {speaker} has now concluded. "
            "For full archives and additional lessons, please visit WSOJ.NET."
        ]
        return random.choice(options)

    # --- Category 2: Medium outro (15–20 minutes) ---
    elif duration_min >= 15:
        options = [
            f"This concludes the lesson {title} by {speaker}.",
            f"You’ve been listening to {title}, presented by {speaker}.",
            f"That was {title} from {speaker}.",
            f"This has been {title}, delivered by {speaker}.",
            f"The lesson {title} by {speaker} has now concluded."
        ]
        return random.choice(options)

    # --- Category 3: Short outro (< 15 minutes) ---
    else:
        options = [
            f"This concludes the lesson by {speaker}.",
            f"You’ve been listening to a message from {speaker}.",
            f"That was a lesson delivered by {speaker}.",
            f"This message from {speaker} has now concluded.",
            f"You’ve just heard a lesson presented by {speaker}."
        ]
        return random.choice(options)
