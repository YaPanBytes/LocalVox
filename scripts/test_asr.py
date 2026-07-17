"""
Standalone ASR test — confirms mic capture, VAD, and faster-whisper
are all working together before wiring in the LLM/TTS stages.

Run: python scripts/test_asr.py
Speak after "Listening...", then pause — it'll transcribe and print.
"""

import sys
import os

# Allow running this script directly from scripts/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio.capture import AudioCapture
from asr.local_whisper import LocalWhisperTranscriber
from utils.timing import timed_stage


def main():
    print("Loading Whisper model (base, GPU)...")
    transcriber = LocalWhisperTranscriber(model_size="base", device="cuda", compute_type="float16")

    print("Setting up mic capture...")
    capture = AudioCapture()

    print("\nListening... speak now, then pause.")
    with timed_stage("capture", print):
        audio = capture.listen_until_silence()

    print(f"Captured {len(audio)} samples ({len(audio)/16000:.2f}s of audio)")

    with timed_stage("transcription", print):
        text = transcriber.transcribe(audio)

    print(f"\nTranscript: \"{text}\"")


if __name__ == "__main__":
    main()