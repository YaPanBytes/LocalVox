"""
State machine driving one conversation turn:
listening -> transcribing -> thinking -> speaking -> idle

Handles offline->online fallback per-stage (ASR/LLM/TTS) and the
"keep the user engaged while slow" fallback via FallbackHandler.
"""

import re
import time
from core.fallback import FallbackHandler
from core.session import Session
from utils.logger import get_logger
from utils.timing import timed_stage

logger = get_logger()

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


class Orchestrator:
    def __init__(self, transcriber, responder, synthesizer, capture, playback,
                 fallback_timeout: float = 1.3):
        self.transcriber = transcriber
        self.responder = responder
        self.synthesizer = synthesizer
        self.capture = capture
        self.playback = playback

        self.session = Session()
        self.fallback = FallbackHandler(playback, timeout_sec=fallback_timeout)

    def run_turn(self):
        """Runs a single listen -> respond -> speak cycle."""

        # --- LISTENING ---
        logger.info("state: listening")
        with timed_stage("capture", logger.info):
            audio = self.capture.listen_until_silence()

        # --- TRANSCRIBING ---
        logger.info("state: transcribing")
        with timed_stage("asr", logger.info):
            user_text = self._transcribe_with_fallback(audio)

        if not user_text:
            logger.info("No speech detected, returning to idle.")
            return

        logger.info(f"User said: {user_text}")
        self.session.add_user_turn(user_text)

        # --- THINKING (with engagement fallback) ---
        logger.info("state: thinking")
        self.fallback.start()
        response_text = ""
        buffer = ""
        thinking_start = time.perf_counter()
        first_sentence_logged = False

        for token in self._respond_with_fallback(user_text):
            buffer += token
            response_text += token

            # As soon as we have a full sentence, stop the filler timer
            # (if it hasn't already fired) and start speaking.
            match = SENTENCE_END_RE.search(buffer)
            if match:
                sentence = buffer[:match.end()].strip()
                buffer = buffer[match.end():]
                self.fallback.stop()
                if not first_sentence_logged:
                    latency = time.perf_counter() - thinking_start
                    logger.info(f"[timing] time to first sentence ready: {latency*1000:.0f}ms")
                    first_sentence_logged = True
                self._speak(sentence)

        if buffer.strip():
            self._speak(buffer.strip())

        total_turn_time = time.perf_counter() - thinking_start
        logger.info(f"[timing] total generation+playback: {total_turn_time*1000:.0f}ms")

        self.fallback.stop()
        self.session.add_assistant_turn(response_text.strip())
        logger.info("state: idle")

    # --- internal helpers ---

    def _transcribe_with_fallback(self, audio) -> str:
        try:
            return self.transcriber.transcribe(audio)
        except Exception as e:
            logger.warning(f"Local ASR failed ({e}), trying online fallback.")
            if self.online_transcriber:
                return self.online_transcriber.transcribe(audio)
            raise

    def _respond_with_fallback(self, user_text: str):
        try:
            yield from self.responder.respond_stream(user_text, self.session.get_history())
        except Exception as e:
            logger.warning(f"Local LLM failed ({e}), trying online fallback.")
            if self.online_responder:
                yield from self.online_responder.respond_stream(user_text, self.session.get_history())
            else:
                raise

    def _speak(self, text: str):
        logger.info(f"state: speaking -> {text}")
        try:
            audio_bytes = self.synthesizer.synthesize(text)
        except Exception as e:
            logger.warning(f"Local TTS failed ({e}), trying online fallback.")
            if self.online_synthesizer:
                audio_bytes = self.online_synthesizer.synthesize(text)
            else:
                raise
        self.playback.play(audio_bytes)