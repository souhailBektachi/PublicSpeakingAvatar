from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "models" / "TTS"

ort.set_default_logger_severity(4)

VOICES_PATH = MODEL_DIR / "voices.bin"
MODEL_PATH = MODEL_DIR / "kokoro-v1.0.onnx"


def get_voices(path: Path = VOICES_PATH) -> list[str]:
    """
    Get the list of available voices without creating a synthesizer instance.
    """
    try:
        voices = np.load(path)
        return list(voices.keys())
    except Exception:
        return []


class SpeechSynthesizer:
    """
    Kokoro-based speech synthesizer for text-to-speech conversion.
    """

    DEFAULT_VOICE: str = "af_alloy"
    MAX_PHONEME_LENGTH: int = 510
    SAMPLE_RATE: int = 24000

    def __init__(self, model_path: Path = MODEL_PATH, voice: str = DEFAULT_VOICE) -> None:
        self.sample_rate = self.SAMPLE_RATE
        # Load voices
        self.voices: dict[str, NDArray[np.float32]] = np.load(VOICES_PATH)
        self.vocab = self._get_vocab()

        self.set_voice(voice)

        available_providers = ort.get_available_providers()
        # Filter out TensorRT to avoid fallback errors, prioritize CUDA
        providers = [p for p in available_providers if p != "TensorrtExecutionProvider"]
        
        print(f"[Kokoro] Active Providers: {providers}")
        
        self.ort_sess = ort.InferenceSession(
            model_path,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )
        
        # Import phonemizer relatively
        from .phonemizer import Phonemizer
        self.phonemizer = Phonemizer(model_dir=MODEL_DIR)

    def set_voice(self, voice: str) -> None:
        if voice not in self.voices:
            # Fallback if voice not found, or use first available
            if self.voices:
                print(f"Voice '{voice}' not found. Falling back to first available.")
                voice = list(self.voices.keys())[0]
            else:
                raise ValueError("No voices available in voices.bin")
        self.voice = voice

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        phonemes = self.phonemizer.convert_to_phonemes([text], "en_us")
        phoneme_ids = self._phonemes_to_ids(phonemes[0])
        audio = self._synthesize_ids_to_audio(phoneme_ids)
        return np.array(audio, dtype=np.float32)

    @staticmethod
    def _get_vocab() -> dict[str, int]:
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = (
            "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻ"
            "ʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        )
        symbols = [_pad, *_punctuation, *_letters, *_letters_ipa]
        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i
        return dicts

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        if len(phonemes) > self.MAX_PHONEME_LENGTH:
             # Truncate if too long to prevent crash, simple workaround
             phonemes = phonemes[:self.MAX_PHONEME_LENGTH]
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def _synthesize_ids_to_audio(self, ids: list[int]) -> NDArray[np.float32]:
        voice_vector = self.voices[self.voice]
        voice_array = voice_vector[len(ids)]

        tokens = [[0, *ids, 0]]
        speed = 1.0
        audio = self.ort_sess.run(
            None,
            {
                "tokens": tokens,
                "style": voice_array,
                "speed": np.ones(1, dtype=np.float32) * speed,
            },
        )[0]
        # Remove silence at end (heuristic from glados repo)
        return np.array(audio[:-8000], dtype=np.float32)

    def __del__(self) -> None:
        if hasattr(self, "ort_sess"):
            del self.ort_sess
