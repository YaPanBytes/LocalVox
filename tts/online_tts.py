"""
Online TTS fallback (e.g. ElevenLabs, OpenAI TTS). Only used if local
synthesis fails or is disabled in config.yaml.
"""

import os
from tts.base import Synthesizer, SynthesisError


class OnlineTTSSynthesizer(Synthesizer):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TTS_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found for online TTS fallback")

    def synthesize(self, text: str) -> bytes:
        try:
            # Placeholder — wire up your online TTS API client here
            raise NotImplementedError("Wire up your online TTS API client here")
        except Exception as e:
            raise SynthesisError(f"Online TTS failed: {e}")
