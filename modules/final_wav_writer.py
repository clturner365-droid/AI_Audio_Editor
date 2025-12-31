"""
final_wav_writer.py

Purpose:
    Save the final assembled AudioSegment to disk,
    then embed metadata into the WAV file as the LAST step.

This module does NOT call any other modules except metadata_writer,
and only after the WAV is written.

It is dispatcher-ready and uses standard logging.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from pydub import AudioSegment

from metadata_writer import prepare_and_write_metadata   # your existing module


@dataclass
class FinalWavWriter:
    logger: Optional[object] = None
    step_id: str = "audio.finalize"
    debug: bool = False

    def write_final_wav(
        self,
        final_audio: AudioSegment,
        output_dir: str,
        base_name: str,
        metadata: dict
    ) -> str:
        """
        Writes the final WAV file, then embeds metadata into it.
        Returns the final WAV path.
        """

        # ---------------------------------------------------------------
        # 1. Ensure output directory exists
        # ---------------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)

        final_path = os.path.join(output_dir, f"{base_name}.wav")

        # ---------------------------------------------------------------
        # 2. Write the WAV file
        # ---------------------------------------------------------------
        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "writing_final_wav",
                "path": final_path
            })

        final_audio.export(final_path, format="wav")

        if self.debug and self.logger:
            self.logger.debug({
                "step": self.step_id,
                "event": "wav_written",
                "path": final_path
            })

        # ---------------------------------------------------------------
        # 3. Embed metadata (LAST STEP)
        # ---------------------------------------------------------------
        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "embedding_metadata",
                "path": final_path
            })

        prepare_and_write_metadata(
            wav_path=final_path,
            metadata=metadata,
            logger=self.logger,
            step_id=f"{self.step_id}.meta",
            debug=self.debug
        )

        if self.logger:
            self.logger.info({
                "step": self.step_id,
                "event": "final_wav_complete",
                "path": final_path
            })

        return final_path
