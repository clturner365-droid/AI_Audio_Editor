from pydub import AudioSegment

def to_segment(x):
    """
    Converts various audio objects into a pydub AudioSegment.
    Supports:
        - AudioSegment
        - Custom objects with raw_data, sample_width, sample_rate, channels
    """
    if x is None:
        return None

    if isinstance(x, AudioSegment):
        return x

    if hasattr(x, "raw_data"):
        return AudioSegment(
            data=x.raw_data,
            sample_width=x.sample_width,
            frame_rate=x.sample_rate,
            channels=x.channels
        )

    raise ValueError("Unsupported audio object type for concatenation")


def assemble_final_audio(intro_audio, sermon_audio, outro_audio, log_buffer=None):
    """
    Assembles the final audio in the order:
        intro (optional) → sermon → outro
    Returns a pydub.AudioSegment.
    """

    if sermon_audio is None:
        raise ValueError("assemble_final_audio: sermon_audio is required")

    if outro_audio is None:
        raise ValueError("assemble_final_audio: outro_audio is required")

    if log_buffer is not None:
        log_buffer.append("Assembling final audio (intro + sermon + outro)")

    intro_seg = to_segment(intro_audio)
    sermon_seg = to_segment(sermon_audio)
    outro_seg = to_segment(outro_audio)

    final_audio = AudioSegment.silent(duration=0)

    if intro_seg is not None:
        final_audio += intro_seg

    final_audio += sermon_seg
    final_audio += outro_seg

    if log_buffer is not None:
        log_buffer.append("Final audio assembly complete")

    return final_audio
