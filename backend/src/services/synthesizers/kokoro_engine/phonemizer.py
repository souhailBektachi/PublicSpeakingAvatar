from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from pickle import load
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)


@dataclass
class ModelConfig:
    MODEL_PATH: Path
    PHONEME_DICT_PATH: Path
    TOKEN_TO_IDX_PATH: Path
    IDX_TO_TOKEN_PATH: Path
    CHAR_REPEATS: int = 3
    MODEL_INPUT_LENGTH: int = 64
    EXPAND_ACRONYMS: bool = False
    USE_CUDA: bool = True

    def __init__(
        self,
        model_dir: Path,
    ) -> None:
        self.MODEL_PATH = model_dir / "phomenizer_en.onnx"
        self.PHONEME_DICT_PATH = model_dir / "lang_phoneme_dict.pkl"
        self.TOKEN_TO_IDX_PATH = model_dir / "token_to_idx.pkl"
        self.IDX_TO_TOKEN_PATH = model_dir / "idx_to_token.pkl"


class SpecialTokens(Enum):
    PAD = "_"
    START = "<start>"
    END = "<end>"
    EN_US = "<en_us>"


class Punctuation(Enum):
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        return re.compile(f"([{cls.PUNCTUATION.value + cls.SPACE.value}])")


class Phonemizer:
    """Phonemizer class for converting text to phonemes."""

    def __init__(self, model_dir: Path) -> None:
        self.config = ModelConfig(model_dir)

        # Load PKL files
        self.phoneme_dict: dict[str, str] = self._load_pickle(self.config.PHONEME_DICT_PATH)
        self.phoneme_dict["glados"] = "ɡlˈɑːdɑːs"  # Keep GLaDOS easter egg/fix

        self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
        self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)

        providers = ort.get_available_providers()
        # Same preference as engine
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.ort_session = ort.InferenceSession(
            self.config.MODEL_PATH,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )

        self.special_tokens: set[str] = {
            SpecialTokens.PAD.value,
            SpecialTokens.END.value,
            SpecialTokens.EN_US.value,
        }

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        with path.open("rb") as f:
            return load(f)  # type: ignore

    @staticmethod
    def _unique_consecutive(arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        result = []
        for row in arr:
            if len(row) == 0:
                result.append(row)
            else:
                mask = np.concatenate(([True], row[1:] != row[:-1]))
                result.append(row[mask])
        return result

    @staticmethod
    def _remove_padding(arr: list[NDArray[np.int64]], padding_value: int = 0) -> list[NDArray[np.int64]]:
        return [row[row != padding_value] for row in arr]

    @staticmethod
    def _trim_to_stop(arr: list[NDArray[np.int64]], end_index: int = 2) -> list[NDArray[np.int64]]:
        result = []
        for row in arr:
            stop_index = np.where(row == end_index)[0]
            if len(stop_index) > 0:
                result.append(row[: stop_index[0] + 1])
            else:
                result.append(row)
        return result

    def _process_model_output(self, arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        arr_processed: list[NDArray[np.int64]] = np.argmax(arr[0], axis=2)
        arr_processed = self._unique_consecutive(arr_processed)
        arr_processed = self._remove_padding(arr_processed)
        arr_processed = self._trim_to_stop(arr_processed)
        return arr_processed

    @staticmethod
    def _expand_acronym(word: str) -> str:
        if Punctuation.HYPHEN.value in word:
            return word
        return word

    def encode(self, sentence: Iterable[str]) -> list[int]:
        try:
             # Ensure CHAR_REPEATS is available via config
             char_repeats = self.config.CHAR_REPEATS
        except AttributeError:
             char_repeats = 3

        sentence = [item for item in sentence for _ in range(char_repeats)]
        sentence = [s.lower() for s in sentence]
        
        # Safe lookup in token_to_idx
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        
        return [
            self.token_to_idx[SpecialTokens.START.value],
            *sequence,
            self.token_to_idx[SpecialTokens.END.value],
        ]

    def decode(self, sequence: NDArray[np.int64]) -> str:
        decoded = []
        for t in sequence:
            idx = t.item()
            token = self.idx_to_token[idx]
            decoded.append(token)
        result = "".join(d for d in decoded if d not in self.special_tokens)
        return result

    @staticmethod
    def pad_sequence_fixed(v: list[list[int]], target_length: int) -> NDArray[np.int64]:
        result: NDArray[np.int64] = np.zeros((len(v), target_length), dtype=np.int64)
        for i, seq in enumerate(v):
            length = min(len(seq), target_length)
            result[i, :length] = seq[:length]
        return result

    def _get_dict_entry(self, word: str, punc_set: set[str]) -> str | None:
        if word in punc_set or len(word) == 0:
            return word
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]
        elif word.lower() in self.phoneme_dict:
            return self.phoneme_dict[word.lower()]
        elif word.title() in self.phoneme_dict:
            return self.phoneme_dict[word.title()]
        else:
            return None

    def _get_phonemes(
        self,
        word: str,
        word_phonemes: dict[str, str | None],
        word_splits: dict[str, list[str]],
    ) -> str:
        phons = word_phonemes[word]
        if phons is None:
            subwords = word_splits[word]
            subphons_converted = [word_phonemes[w] for w in subwords]
            # Filter None and join
            phons = "".join([subphon for subphon in subphons_converted if subphon is not None])
        return phons

    def _clean_and_split_texts(
        self, texts: list[str], punc_set: set[str], punc_pattern: re.Pattern[str]
    ) -> tuple[list[list[str]], set[str]]:
        split_text, cleaned_words = [], set[str]()
        for text in texts:
            cleaned_text = "".join(t for t in text if t.isalnum() or t in punc_set)
            split = [s for s in re.split(punc_pattern, cleaned_text) if len(s) > 0]
            split_text.append(split)
            cleaned_words.update(split)
        return split_text, cleaned_words

    def convert_to_phonemes(self, texts: list[str], lang: str = "en_us") -> list[str]:
        split_text: list[list[str]] = []
        cleaned_words = set[str]()

        punc_set = Punctuation.get_punc_set()
        punc_pattern = Punctuation.get_punc_pattern()

        split_text, cleaned_words = self._clean_and_split_texts(texts, punc_set, punc_pattern)

        for punct in punc_set:
            self.phoneme_dict[punct] = punct
        word_phonemes = {word: self.phoneme_dict.get(word.lower()) for word in cleaned_words}

        words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]

        word_splits = {
            key: re.split(
                r"([-])",
                self._expand_acronym(word) if self.config.EXPAND_ACRONYMS else word,
            )
            for key, word in zip(words_to_split, words_to_split, strict=False)
        }

        subwords = {w for values in word_splits.values() for w in values if w not in word_phonemes}

        for subword in subwords:
            word_phonemes[subword] = self._get_dict_entry(word=subword, punc_set=punc_set)

        words_to_predict = [
            word for word, phons in word_phonemes.items() if phons is None and len(word_splits.get(word, [])) <= 1
        ]

        if words_to_predict:
            input_batch = [self.encode(word) for word in words_to_predict]
            input_batch_padded: NDArray[np.int64] = self.pad_sequence_fixed(input_batch, self.config.MODEL_INPUT_LENGTH)

            ort_inputs = {self.ort_session.get_inputs()[0].name: input_batch_padded}
            ort_outs = self.ort_session.run(None, ort_inputs)

            ids = self._process_model_output(ort_outs)

            for id, word in zip(ids, words_to_predict, strict=False):
                word_phonemes[word] = self.decode(id)

        phoneme_lists = []
        for text in split_text:
            text_phons = [
                self._get_phonemes(word=word, word_phonemes=word_phonemes, word_splits=word_splits) for word in text
            ]
            phoneme_lists.append(text_phons)

        return ["".join(phoneme_list) for phoneme_list in phoneme_lists]

    def __del__(self) -> None:
        if hasattr(self, "ort_sess"):
            del self.ort_sess
