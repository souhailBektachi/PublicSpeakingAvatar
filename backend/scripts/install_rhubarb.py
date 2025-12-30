"""
Download and install Rhubarb Lip Sync CLI.
Run this script to set up lip sync functionality for the avatar.

Usage: python scripts/install_rhubarb.py
"""
import os
import sys
import platform
import zipfile
import tarfile
import shutil
import urllib.request
from pathlib import Path

# Rhubarb release information
RHUBARB_VERSION = "1.13.0"
RHUBARB_BASE_URL = f"https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v{RHUBARB_VERSION}"

# Installation directory (relative to backend root)
INSTALL_DIR = Path(__file__).parent.parent / "bin" / "rhubarb"


def get_download_url() -> str:
    """Get the appropriate download URL for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return f"{RHUBARB_BASE_URL}/Rhubarb-Lip-Sync-{RHUBARB_VERSION}-Windows.zip"
    elif system == "darwin":  # macOS
        return f"{RHUBARB_BASE_URL}/Rhubarb-Lip-Sync-{RHUBARB_VERSION}-macOS.zip"
    elif system == "linux":
        return f"{RHUBARB_BASE_URL}/Rhubarb-Lip-Sync-{RHUBARB_VERSION}-Linux.zip"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress indication."""
    print(f"Downloading from: {url}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
        sys.stdout.write(f"\rProgress: {percent:.1f}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()  # New line after progress


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a zip or tar archive."""
    print(f"Extracting to: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(dest_dir)
    else:
        raise RuntimeError(f"Unknown archive format: {archive_path}")


def find_rhubarb_executable(extract_dir: Path) -> Path:
    """Find the rhubarb executable in the extracted directory."""
    exe_name = "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"
    
    for root, dirs, files in os.walk(extract_dir):
        if exe_name in files:
            return Path(root) / exe_name
    
    raise RuntimeError(f"Could not find {exe_name} in extracted files")


def install_rhubarb() -> Path:
    """Download and install Rhubarb Lip Sync CLI."""
    print("=" * 50)
    print("Rhubarb Lip Sync Installer")
    print("=" * 50)
    
    # Check if already installed
    exe_name = "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"
    installed_path = INSTALL_DIR / exe_name
    
    if installed_path.exists():
        print(f"Rhubarb already installed at: {installed_path}")
        return installed_path
    
    # Create temp directory for download
    temp_dir = Path(__file__).parent.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Download
        url = get_download_url()
        archive_name = url.split("/")[-1]
        archive_path = temp_dir / archive_name
        
        if not archive_path.exists():
            download_file(url, archive_path)
        else:
            print(f"Using cached download: {archive_path}")
        
        # Extract
        extract_dir = temp_dir / "rhubarb_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_archive(archive_path, extract_dir)
        
        # Find and move executable
        exe_path = find_rhubarb_executable(extract_dir)
        INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire Rhubarb directory (includes res/ folder needed for phoneme recognition)
        rhubarb_content_dir = exe_path.parent
        for item in rhubarb_content_dir.iterdir():
            dest = INSTALL_DIR / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        
        # Make executable on Unix
        if platform.system() != "Windows":
            os.chmod(installed_path, 0o755)
        
        print(f"\n✓ Rhubarb installed successfully at: {installed_path}")
        return installed_path
        
    finally:
        # Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    try:
        path = install_rhubarb()
        
        # Test the installation
        print("\nTesting installation...")
        import subprocess
        result = subprocess.run([str(path), "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Rhubarb version: {result.stdout.strip()}")
            print("\n✓ Installation complete! Lip sync is ready to use.")
        else:
            print(f"✗ Test failed: {result.stderr}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
