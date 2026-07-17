"""
Standalone LLM test — confirms Ollama server + model + streaming
response all work before wiring into the full pipeline.

Run: python scripts/test_llm.py
Type a message, see the streamed response and timing.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm.local_ollama import LocalOllamaResponder
from utils.timing import timed_stage


def main():
    print("Connecting to Ollama (llama3.2:1b)...")
    responder = LocalOllamaResponder(
        model="llama3.2:1b",
        system_prompt="You are a helpful, concise voice assistant. Keep answers short and natural for speech.",
    )

    history = []

    print("\nType a message and press Enter (Ctrl+C to exit).\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user_text:
            continue

        print("Assistant: ", end="", flush=True)
        full_response = ""

        with timed_stage("\n[llm total]", print):
            first_token_time = None
            import time
            start = time.perf_counter()

            for token in responder.respond_stream(user_text, history):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                print(token, end="", flush=True)
                full_response += token

        if first_token_time is not None:
            print(f"[timing] time to first token: {first_token_time*1000:.0f}ms")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": full_response.strip()})


if __name__ == "__main__":
    main()