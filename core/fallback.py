"""
Detects when the LLM/ASR is taking too long and plays a pre-generated
filler clip to keep the user engaged instead of dead air or an error.
"""

import os
import random
import threading


class FallbackHandler:
    def __init__(self, playback, fillers_dir: str = "assets/fillers", timeout_sec: float = 1.3):
        self.playback = playback
        self.timeout_sec = timeout_sec
        self.fillers_dir = fillers_dir
        self._timer = None
        self._triggered = False

    def _load_filler_clips(self):
        if not os.path.isdir(self.fillers_dir):
            return []
        return [os.path.join(self.fillers_dir, f) for f in os.listdir(self.fillers_dir) if f.endswith(".raw")]

    def _play_random_filler(self):
        clips = self._load_filler_clips()
        if not clips:
            return
        chosen = random.choice(clips)
        with open(chosen, "rb") as f:
            audio_bytes = f.read()
        self._triggered = True
        self.playback.play(audio_bytes)

    def start(self):
        """Call this right before starting the LLM call. Cancel with stop() once a response arrives."""
        self._triggered = False
        self._timer = threading.Timer(self.timeout_sec, self._play_random_filler)
        self._timer.start()

    def stop(self):
        """Call this as soon as the real response is ready to play."""
        if self._timer:
            self._timer.cancel()

    def was_triggered(self) -> bool:
        return self._triggered