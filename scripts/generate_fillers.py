"""
Generates fallback filler clips using Piper, saved as raw PCM (no WAV
header) to match what playback.py expects — same format Piper's
--output_raw produces, which is what local_piper.py uses at runtime.

Run: python scripts/generate_fillers.py
"""

import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PIPER_EXE = "piper/piper.exe"
VOICE_MODEL = "models/piper_voices/en_US-lessac-medium.onnx"
OUTPUT_DIR = "assets/fillers"

FILLER_PHRASES = [
    "Let me think about that for a second.",
    "Good question, one moment.",
    "Hmm, give me just a second.",
    "Let me look into that.",
    "One moment, thinking it through.",
]


def generate_filler(text: str, output_path: str):
    result = subprocess.run(
        [PIPER_EXE, "--model", VOICE_MODEL, "--output_raw"],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    with open(output_path, "wb") as f:
        f.write(result.stdout)
    print(f"Generated: {output_path} ({len(result.stdout)} bytes)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, phrase in enumerate(FILLER_PHRASES, start=1):
        output_path = os.path.join(OUTPUT_DIR, f"filler_{i:02d}.raw")
        generate_filler(phrase, output_path)
    print(f"\nDone. {len(FILLER_PHRASES)} filler clips saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()