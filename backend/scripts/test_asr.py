import sys
import os
import numpy as np
import time
from pathlib import Path
import soundfile as sf

backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_root)

from src.services.parakeet_asr import ParakeetASR
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Parakeet ASR...")
    try:
        asr = ParakeetASR()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    print("Model loaded successfully.")
    
    data_dir = Path(backend_root) / "assets" / "audio" / "samples"
    wav_files = list(data_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {data_dir}")
        return

    print(f"Found {len(wav_files)} files in {data_dir}")
    
    for wav_path in wav_files:
        print(f"\nProcessing {wav_path.name}...")
        try:
            audio, sr = sf.read(str(wav_path), dtype='float32')
            
            if sr != 16000:
                print(f"  Note: Resampling from {sr} to 16000")
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            start = time.time()
            text = asr.transcribe(audio)
            duration = time.time() - start
            
            print(f"  Result: '{text}'")
            print(f"  Inference took: {duration:.2f}s")
            
        except Exception as e:
            print(f"  Failed to process {wav_path.name}: {e}")

    print("\nTest complete.")

if __name__ == "__main__":
    main()
