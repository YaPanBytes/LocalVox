"""
Microphone capture with VAD (voice activity detection) to detect
end-of-speech instead of relying on a fixed silence timeout.
"""

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, vad_aggressiveness: int = 2):
        if sd is None:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")
        if webrtcvad is None:
            raise ImportError("webrtcvad not installed. Run: pip install webrtcvad")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    def listen_until_silence(self, max_silence_frames: int = 20) -> np.ndarray:
        """
        Record audio from the mic until VAD detects sustained silence,
        signaling the user has finished speaking. Returns a float32
        numpy array of the captured speech.
        """
        frames = []
        silence_count = 0
        speech_started = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16") as stream:
            while True:
                frame, _ = stream.read(self.frame_size)
                frame_bytes = frame.tobytes()
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

                if is_speech:
                    speech_started = True
                    silence_count = 0
                    frames.append(frame)
                elif speech_started:
                    silence_count += 1
                    frames.append(frame)
                    if silence_count > max_silence_frames:
                        break

        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        audio = audio.flatten() 
        return audio
