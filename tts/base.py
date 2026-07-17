"""
Synthesizer interface — both local (offline) and online implementations
must conform to this.
"""

from abc import ABC, abstractmethod


class Synthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Take a piece of text (typically one sentence) and return raw
        audio bytes ready for playback.
        """
        raise NotImplementedError


class SynthesisError(Exception):
    """Raised when synthesis fails."""
    pass
