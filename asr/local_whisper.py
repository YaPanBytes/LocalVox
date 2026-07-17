"""
Offline ASR using faster-whisper. Runs on your MX450 GPU (int8/float16)
or falls back to CPU automatically if no GPU is available.
"""

from asr.base import Transcriber, TranscriptionError

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class LocalWhisperTranscriber(Transcriber):
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
        if WhisperModel is None:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            # faster-whisper expects a file path or numpy array/generator;
            # in practice you'll pass a numpy float32 array from audio/capture.py
            segments, _info = self.model.transcribe(audio_bytes, beam_size=1)
            text = " ".join(segment.text for segment in segments).strip()
            return text  # empty string is valid - means no speech detected, not an error
        except Exception as e:
            raise TranscriptionError(f"Local whisper failed: {e}")