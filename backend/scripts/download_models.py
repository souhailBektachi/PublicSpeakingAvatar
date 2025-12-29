import os
import hashlib
import urllib.request
from pathlib import Path
import sys

MODEL_DETAILS = {
    "models/ASR/parakeet-tdt-0.6b-v2_encoder.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_encoder.onnx",
        "checksum": "f133a92186e63c7d4ab5b395a8e45d49f4a7a84a1d80b66f494e8205dfd57b63",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_decoder.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_decoder.onnx",
        "checksum": "415b14f965b2eb9d4b0b8517f0a1bf44a014351dd43a09c3a04d26a41c877951",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_joiner.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_joiner.onnx",
        "checksum": "846929b668a94462f21be25c7b5a2d83526e0b92a8306f21d8e336fc98177976",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_model_config.yaml": {
        "url": "https://raw.githubusercontent.com/dnhkng/GlaDOS/main/models/ASR/parakeet-tdt-0.6b-v2_model_config.yaml",
        "checksum": None 
    },
}

def download_file(url, file_path, expected_checksum=None):
    print(f"Downloading {file_path}...")
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading... {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, file_path, reporthook)
        print("\nDownload complete.")
        
        if expected_checksum:
            print("Verifying checksum...")
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            calculated_checksum = sha256_hash.hexdigest()
            if calculated_checksum != expected_checksum:
                print(f"Error: Checksum mismatch for {file_path}")
                print(f"Expected: {expected_checksum}")
                print(f"Got:      {calculated_checksum}")
                os.remove(file_path)
                return False
            print("Checksum verified.")
        return True

    except Exception as e:
        print(f"Error downloading {file_path}: {e}")
        return False

def main():
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")
    
    success = True
    for relative_path, info in MODEL_DETAILS.items():
        dest_path = project_root / relative_path
        if dest_path.exists():
            if info["checksum"]:
                print(f"Verifying existing file {relative_path}...")
                sha256_hash = hashlib.sha256()
                with open(dest_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
                if sha256_hash.hexdigest() == info["checksum"]:
                    print(f"File {relative_path} already exists and is valid. Skipping.")
                    continue
                else:
                    print(f"File {relative_path} exists but has wrong checksum. Re-downloading.")
            else:
                print(f"File {relative_path} already exists. Skipping (no checksum).")
                continue
                
        if not download_file(info["url"], dest_path, info["checksum"]):
            success = False
            
    if success:
        print("\nAll models downloaded successfully.")
        sys.exit(0)
    else:
        print("\nSome downloads failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
