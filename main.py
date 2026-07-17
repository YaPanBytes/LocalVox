"""
Entry point. Run: python main.py

Simple local loop, no web server — press Ctrl+C to exit.
"""

import yaml
from core.orchestrator import Orchestrator
from audio.capture import AudioCapture
from audio.playback import AudioPlayback
from utils.logger import get_logger

from asr.local_whisper import LocalWhisperTranscriber
from llm.local_ollama import LocalOllamaResponder
from tts.local_piper import LocalPiperSynthesizer

logger = get_logger()


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml not found, using defaults.")
        return {}




def main():
    config = load_config()

    capture = AudioCapture()
    playback = AudioPlayback()

    transcriber = LocalWhisperTranscriber(
        model_size=config.get("whisper_model", "base"),
        device=config.get("whisper_device", "cuda"),
    )
    responder = LocalOllamaResponder(
        model=config.get("llm_model", "llama3.2:1b"),
        system_prompt=config.get("system_prompt", "You are a helpful, concise voice assistant."),
    )
    synthesizer = LocalPiperSynthesizer(
        voice_model_path=config.get("piper_voice_path", "models/piper_voices/en_US-lessac-medium.onnx"),
        piper_exe_path=config.get("piper_exe_path", "piper/piper.exe"),
    )

    orchestrator = Orchestrator(
        transcriber=transcriber,
        responder=responder,
        synthesizer=synthesizer,
        capture=capture,
        playback=playback,
        fallback_timeout=config.get("fallback_timeout_sec", 1.3),
    )

    logger.info("Voice assistant ready. Speak after the prompt (Ctrl+C to exit).")
    try:
        while True:
            try:
                orchestrator.run_turn()
            except Exception as e:
                logger.error(f"Turn failed ({e}), returning to listening.")
    except KeyboardInterrupt:
        logger.info("Exiting.")


if __name__ == "__main__":
    main()