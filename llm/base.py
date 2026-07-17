"""
Responder interface — both local (offline) and online implementations
must conform to this. respond_stream() is what enables sentence-by-sentence
TTS streaming instead of waiting for the full response.
"""

from abc import ABC, abstractmethod
from typing import Iterator


class Responder(ABC):
    @abstractmethod
    def respond_stream(self, user_text: str, history: list) -> Iterator[str]:
        """
        Yield response text in chunks (tokens or small groups of tokens)
        as they're generated, so the caller can start TTS on the first
        completed sentence rather than waiting for the full reply.
        """
        raise NotImplementedError


class ResponseError(Exception):
    """Raised when the LLM fails or times out."""
    pass
