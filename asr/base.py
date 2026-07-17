"""
Transcriber interface — both local (offline) and online implementations
must conform to this so the orchestrator never cares which one is active.
"""

from abc import ABC, abstractmethod


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Take raw audio (e.g. 16kHz mono PCM) and return the transcribed text.
        Should raise TranscriptionError on failure so orchestrator can
        decide whether to fall back to the online implementation.
        """
        raise NotImplementedError


class TranscriptionError(Exception):
    """Raised when transcription fails or times out."""
    pass
