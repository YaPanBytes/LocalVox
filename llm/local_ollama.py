"""
Offline LLM using Ollama (e.g. llama3.2:3b, phi3.5). Runs on CPU by
default given MX450's limited VRAM — see config.yaml to change model.
"""

from typing import Iterator
from llm.base import Responder, ResponseError

try:
    import ollama
except ImportError:
    ollama = None


class LocalOllamaResponder(Responder):
    def __init__(self, model: str = "llama3.2:1b", system_prompt: str = ""):
        if ollama is None:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        self.model = model
        self.system_prompt = system_prompt

    def respond_stream(self, user_text: str, history: list) -> Iterator[str]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        try:
            stream = ollama.chat(model=self.model, messages=messages, stream=True)
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception as e:
            raise ResponseError(f"Local LLM failed: {e}")
