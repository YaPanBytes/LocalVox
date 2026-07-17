"""
Online ASR fallback (e.g. OpenAI Whisper API). Only used if local
transcription fails or is disabled in config.yaml.
"""

import os
from asr.base import Transcriber, TranscriptionError


class OnlineWhisperTranscriber(Transcriber):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found for online ASR fallback")

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            # Placeholder — wire up actual API client here
            # e.g. openai.Audio.transcribe("whisper-1", audio_file)
            raise NotImplementedError("Wire up your online ASR API client here")
        except Exception as e:
            raise TranscriptionError(f"Online whisper failed: {e}")
