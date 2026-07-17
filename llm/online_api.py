"""
Online LLM fallback (any hosted API — Anthropic, OpenAI, etc).
Only used if local generation fails or is disabled in config.yaml.
"""

import os
from typing import Iterator
from llm.base import Responder, ResponseError


class OnlineAPIResponder(Responder):
    def __init__(self, api_key: str = None, model: str = "your-hosted-model"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("No API key found for online LLM fallback")

    def respond_stream(self, user_text: str, history: list) -> Iterator[str]:
        try:
            # Placeholder — wire up your streaming API client here
            raise NotImplementedError("Wire up your online LLM API client here")
        except Exception as e:
            raise ResponseError(f"Online LLM failed: {e}")
