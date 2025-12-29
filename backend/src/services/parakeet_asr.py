from pathlib import Path
import os
import time
import typing
import logging
import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort 
import soundfile as sf
import yaml

from src.utils.mel_spectrogram import MelSpectrogramCalculator, MelSpectrogramConfig

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance
_INSTANCE: typing.Optional["ParakeetASR"] = None

def get_asr_instance() -> "ParakeetASR":
    """
    Get or create the singleton instance of ParakeetASR.
    """
    global _INSTANCE
    if _INSTANCE is None:
        logger.info("Initializing ParakeetASR singleton...")
        _INSTANCE = ParakeetASR()
    return _INSTANCE

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)

def _fix_ort_cuda_path():
    """
    On Windows, onnxruntime-gpu often fails to find CUDA DLLs (like cublasLt64_12.dll)
    even if they are installed via torch. This helper finds the torch/lib directory
    and adds it to the system PATH and DLL search path.
    """
    if os.name != 'nt':
        return

    try:
        import torch
        torch_lib_path = Path(torch.__file__).parent / "lib"
        if torch_lib_path.exists():
            path_str = str(torch_lib_path)
            if path_str not in os.environ.get("PATH", ""):
                logger.info(f"Adding {path_str} to PATH to satisfy CUDA dependencies.")
                os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
            
            # For Python 3.8+, we also need to use os.add_dll_directory
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path_str)
                    logger.debug(f"Added {path_str} to DLL directory.")
                except Exception as e:
                    logger.warning(f"Failed to add {path_str} to DLL directory: {e}")
    except ImportError:
        logger.debug("Torch not found, skipping CUDA path injection.")
    except Exception as e:
        logger.warning(f"Error while fixing CUDA path: {e}")

# Apply the fix immediately
_fix_ort_cuda_path()

def resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / relative_path

class _OnnxTDTModel:
    """
    Internal helper class to manage the three ONNX sessions (Encoder, Decoder, Joiner)
    for the TDT ASR model and related metadata.
    """

    DEFAULT_ENCODER_MODEL_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_encoder.onnx")
    DEFAULT_DECODER_MODEL_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_decoder.onnx")
    DEFAULT_JOINER_MODEL_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_joiner.onnx")

    def __init__(
        self,
        providers: list[str],
        encoder_model_path: Path = DEFAULT_ENCODER_MODEL_PATH,
        decoder_model_path: Path = DEFAULT_DECODER_MODEL_PATH,
        joiner_model_path: Path = DEFAULT_JOINER_MODEL_PATH,
    ) -> None:
        """
        Initializes the ONNX model sessions and extracts necessary metadata.
        """
        session_opts = ort.SessionOptions()

        # Enable memory pattern optimization for potential speedup
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.enable_mem_pattern = True

        logger.info(f"Using ONNX providers: {providers}")
        self.encoder = self._init_session(encoder_model_path, session_opts, providers)
        self.decoder = self._init_session(decoder_model_path, session_opts, providers)
        self.joiner = self._init_session(joiner_model_path, session_opts, providers)

        logger.info(f"Encoder active providers: {self.encoder.get_providers()}")
        logger.info(f"Decoder active providers: {self.decoder.get_providers()}")
        logger.info(f"Joiner active providers: {self.joiner.get_providers()}")

        # Extract metadata from encoder
        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        self.normalize_type: str | None = encoder_meta.get("normalize_type")
        self.pred_rnn_layers = int(encoder_meta.get("pred_rnn_layers", 0))
        self.pred_hidden = int(encoder_meta.get("pred_hidden", 0))
        logger.info(f"Encoder metadata: {encoder_meta}")
        
        if not self.pred_rnn_layers or not self.pred_hidden:
            logger.warning(
                "Warning: Could not extract 'pred_rnn_layers' or 'pred_hidden' from encoder metadata. "
                "Decoder state initialization might fail."
            )

        # Get joiner output dimension
        self.joiner_output_total_dim = self.joiner.get_outputs()[0].shape[-1]
        
        # Store input/output names
        self.encoder_in_names = [i.name for i in self.encoder.get_inputs()]
        self.encoder_out_names = [o.name for o in self.encoder.get_outputs()]
        self.decoder_in_names = [i.name for i in self.decoder.get_inputs()]
        self.decoder_out_names = [o.name for o in self.decoder.get_outputs()]
        self.joiner_in_names = [i.name for i in self.joiner.get_inputs()]
        self.joiner_out_names = [o.name for o in self.joiner.get_outputs()]

    def _init_session(
        self, model_path: Path, sess_options: ort.SessionOptions, providers: list[str]
    ) -> ort.InferenceSession:
        try:
            return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX session for {model_path}: {e}") from e

    def get_decoder_initial_state(self, batch_size: int = 1) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not self.pred_rnn_layers or not self.pred_hidden:
            raise ValueError("Cannot create decoder state. Missing 'pred_rnn_layers' or 'pred_hidden' metadata.")

        dtype = np.float32
        state0 = np.zeros((self.pred_rnn_layers, batch_size, self.pred_hidden), dtype=dtype)
        state1 = np.zeros((self.pred_rnn_layers, batch_size, self.pred_hidden), dtype=dtype)
        return state0, state1

    def run_encoder(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        feature_length = np.array([features.shape[2]], dtype=np.int64)
        if len(self.encoder_in_names) != 2:
            raise ValueError(f"Encoder expected 2 inputs, got {len(self.encoder_in_names)}")

        input_dict = {
            self.encoder_in_names[0]: features,
            self.encoder_in_names[1]: feature_length,
        }
        encoder_out = self.encoder.run([self.encoder_out_names[0]], input_dict)[0]
        return np.asarray(encoder_out, dtype=np.float32)

    def run_decoder(
        self, token_input: int, state0: NDArray[np.float32], state1: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        target = np.array([[token_input]], dtype=np.int32)
        target_len = np.array([1], dtype=np.int32)
        state0_fp32 = state0.astype(np.float32)
        state1_fp32 = state1.astype(np.float32)

        input_dict = {
            self.decoder_in_names[0]: target,
            self.decoder_in_names[1]: target_len,
            self.decoder_in_names[2]: state0_fp32,
            self.decoder_in_names[3]: state1_fp32,
        }

        outputs = self.decoder.run(self.decoder_out_names, input_dict)
        decoder_out = outputs[0]
        next_state0 = outputs[2]
        next_state1 = outputs[3]

        return decoder_out, next_state0, next_state1

    def run_joiner(self, encoder_out_t: NDArray[np.float32], decoder_out: NDArray[np.float32]) -> NDArray[np.float32]:
        input_dict = {
            self.joiner_in_names[0]: encoder_out_t,
            self.joiner_in_names[1]: decoder_out,
        }
        logits = self.joiner.run(self.joiner_out_names, input_dict)[0]
        return np.asarray(logits, dtype=np.float32)

    def __del__(self) -> None:
        try:
            del self.encoder
            del self.decoder
            del self.joiner
        except AttributeError:
            pass


class ParakeetASR:
    """
    Transcribes audio using a TDT (Token and Duration Transducer) ASR model.
    """

    DEFAULT_CONFIG_PATH = resource_path("models/ASR/parakeet-tdt-0.6b-v2_model_config.yaml")

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.config: dict[str, typing.Any]
        if not config_path.exists():
            raise FileNotFoundError(f"Main YAML configuration file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {config_path}: {e}") from e

        # Configure ONNX Runtime session
        providers = ort.get_available_providers()
        # Optimization: Use CUDA if available
        if "CUDAExecutionProvider" in providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.model = _OnnxTDTModel(providers)

        if "labels" not in self.config:
            raise ValueError("YAML missing 'labels' section for vocabulary configuration.")
        self.idx2token: dict[int, str] = dict(enumerate(self.config["labels"]))
        num_tokens = len(self.idx2token)
        
        # Add blank token to vocab
        self.blank_id = num_tokens
        self.idx2token[self.blank_id] = "<blank>"

        self.tdt_durations = self.config["model_defaults"]["tdt_durations"]
        if not self.tdt_durations:
            raise ValueError("TDT durations list is empty in the configuration.")

        # Initialize Mel Spectrogram calculator from config
        preprocessor_conf_dict = self.config["preprocessor"]
        self.preprocessor_conf = MelSpectrogramConfig(**preprocessor_conf_dict)
        self.melspectrogram = MelSpectrogramCalculator.from_config(self.preprocessor_conf)
        
        logger.info("ParakeetASR initialized successfully.")

    def _process_audio(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        mel_spec = self.melspectrogram.compute(audio)
        mel_spec = np.expand_dims(mel_spec, axis=0)  # Shape: [1, n_mels, time]
        return mel_spec.astype(np.float32)

    def _decode_tdt(self, encoder_out: NDArray[np.float32]) -> list[int]:
        batch_size, _, max_encoder_t = encoder_out.shape
        predicted_token_ids: list[int] = []
        last_emitted_token_for_decoder = self.blank_id
        state0, state1 = self.model.get_decoder_initial_state(batch_size=1)

        decoder_out, next_state0, next_state1 = self.model.run_decoder(last_emitted_token_for_decoder, state0, state1)

        current_t = 0
        max_steps = max_encoder_t * 2
        steps_taken = 0

        while current_t < max_encoder_t and steps_taken < max_steps:
            steps_taken += 1
            encoder_out_t = encoder_out[:, :, current_t : current_t + 1]
            
            joiner_logits = self.model.run_joiner(encoder_out_t, decoder_out)
            joiner_logits = joiner_logits.squeeze()

            token_logits = joiner_logits[: self.blank_id + 1]
            duration_logits = joiner_logits[self.blank_id + 1 :]

            predicted_token_idx = np.argmax(token_logits)
            predicted_duration_bin_idx = np.argmax(duration_logits)
            predicted_skip_amount = self.tdt_durations[predicted_duration_bin_idx]

            if predicted_token_idx != self.blank_id:
                predicted_token_ids.append(int(predicted_token_idx))
                last_emitted_token_for_decoder = int(predicted_token_idx)
                state0 = next_state0
                state1 = next_state1
                decoder_out, next_state0, next_state1 = self.model.run_decoder(
                    last_emitted_token_for_decoder, state0, state1
                )

            current_t += predicted_skip_amount

        return predicted_token_ids

    def _post_process_text(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        tokens_str_list = [self.idx2token.get(idx, "") for idx in token_ids]
        underline = "▁"
        text = "".join(tokens_str_list).replace(underline, " ").strip()
        return text

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribes audio data (numpy array, float32, 16kHz mono).
        """
        start_time = time.time()
        
        # Preprocess
        features = self._process_audio(audio)
        
        # Encoder
        encoder_out = self.model.run_encoder(features)
        
        # Decoding
        predicted_token_ids = self._decode_tdt(encoder_out)
        
        # Post-process
        text = self._post_process_text(predicted_token_ids)
        
        total_time = time.time() - start_time
        logger.info(f"Transcription complete in {total_time:.3f}s: '{text}'")
        
        return text
