"""
Offline TTS using Piper. Lightweight and fast enough to run on CPU,
keeping the MX450 free for ASR.
"""

import subprocess
from tts.base import Synthesizer, SynthesisError


class LocalPiperSynthesizer(Synthesizer):
    def __init__(self,
        voice_model_path: str = "models/piper_voices/en_US-lessac-medium.onnx",
        piper_exe_path: str = "piper/piper.exe"):
        self.voice_model_path = voice_model_path
        self.piper_exe_path = piper_exe_path

    def synthesize(self, text: str) -> bytes:
        try:
            result = subprocess.run(
            [self.piper_exe_path, "--model", self.voice_model_path, "--output_raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
            return result.stdout
        except Exception as e:
            raise SynthesisError(f"Local Piper failed: {e}")
