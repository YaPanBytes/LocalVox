"""
Streams synthesized audio to the speakers as chunks arrive, rather
than waiting for the entire response to be synthesized first.
"""

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    sd = None


class AudioPlayback:
    def __init__(self, sample_rate: int = 22050):
        if sd is None:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")
        self.sample_rate = sample_rate

    def play(self, audio_bytes: bytes):
        """Play a chunk of raw audio (e.g. one sentence) immediately."""
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        sd.play(audio_array, samplerate=self.sample_rate)
        sd.wait()
