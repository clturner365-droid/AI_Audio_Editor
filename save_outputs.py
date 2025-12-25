# modules/save_outputs.py
# Save processed audio, transcript, and metadata to disk.

import os
import json
import soundfile as sf
from pathlib import Path
from modules.logging_system import append_file_log


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_output_files(output_dir: str, base_name: str,
                      waveform, sr: int,
                      transcript: str,
                      metadata: dict,
                      log_buffer):
    """
    Saves:
      - WAV file (16-bit PCM)
      - transcript (.txt)
      - metadata (.json)

    Filenames:
      {output_dir}/{base_name}.wav
      {output_dir}/{base_name}.txt
      {output_dir}/{base_name}.json
    """

    append_file_log(log_buffer, f"Saving outputs to {output_dir} with base name {base_name}...")

    try:
        _ensure_dir(output_dir)

        wav_path = os.path.join(output_dir, f"{base_name}.wav")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        json_path = os.path.join(output_dir, f"{base_name}.json")

        # Save WAV as 16-bit PCM
        try:
            sf.write(wav_path, waveform, sr, subtype="PCM_16")
            append_file_log(log_buffer, f"WAV saved: {wav_path}")
        except Exception as e:
            append_file_log(log_buffer, f"Failed to save WAV: {e}")

        # Save transcript
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript or "")
            append_file_log(log_buffer, f"Transcript saved: {txt_path}")
        except Exception as e:
            append_file_log(log_buffer, f"Failed to save transcript: {e}")

        # Save metadata (merge with existing keys)
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata or {}, f, indent=2)
            append_file_log(log_buffer, f"Metadata saved: {json_path}")
        except Exception as e:
            append_file_log(log_buffer, f"Failed to save metadata: {e}")

        return {"wav": wav_path, "transcript": txt_path, "metadata": json_path}

    except Exception as e:
        append_file_log(log_buffer, f"save_output_files failed: {e}")
        return {}
