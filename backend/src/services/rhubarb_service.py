"""
Rhubarb Lip Sync service for extracting viseme data from audio.
Uses Rhubarb CLI to generate phoneme timings for lip sync animation.
"""
import json
import logging
import subprocess
import tempfile
import base64
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Viseme:
    """A single viseme with timing information."""
    time: float
    shape: str  # A, B, C, D, E, F, G, H, X

@dataclass
class LipSyncResult:
    """Result of lip sync analysis."""
    visemes: List[Viseme]
    duration: float

class RhubarbService:
    """
    Service for extracting viseme data from audio using Rhubarb Lip Sync CLI.
    
    Rhubarb maps audio to mouth shapes (A-H, X for silence).
    These can be mapped to avatar blend shapes for lip sync animation.
    """
    
    # Path to Rhubarb executable - check local bin first, then system PATH
    _LOCAL_RHUBARB_DIR = Path(__file__).parent.parent.parent / "bin" / "rhubarb"
    
    @classmethod
    def _get_rhubarb_path(cls) -> str:
        """Get the path to Rhubarb executable, preferring local installation."""
        import platform
        exe_name = "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"
        local_exe = cls._LOCAL_RHUBARB_DIR / exe_name
        
        if local_exe.exists():
            return str(local_exe)
        return "rhubarb"  # Fall back to system PATH
    
    def __init__(self, rhubarb_path: Optional[str] = None):
        self.rhubarb_path = rhubarb_path or self._get_rhubarb_path()
        self._check_installation()
    
    def _check_installation(self) -> bool:
        """Check if Rhubarb CLI is available."""
        try:
            result = subprocess.run(
                [self.rhubarb_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Rhubarb Lip Sync available: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.warning("Rhubarb CLI not found in PATH. Lip sync will be disabled.")
        except Exception as e:
            logger.warning(f"Could not verify Rhubarb installation: {e}")
        return False
    
    def extract_visemes(self, audio_base64: str, audio_format: str = "mp3") -> Optional[LipSyncResult]:
        """
        Extract viseme data from base64-encoded audio.
        
        Args:
            audio_base64: Base64-encoded audio data
            audio_format: Audio format (mp3, wav, etc.)
            
        Returns:
            LipSyncResult with viseme timings, or None on failure
        """
        try:
            # Decode audio to temp file
            audio_data = base64.b64decode(audio_base64)
            
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as audio_file:
                audio_file.write(audio_data)
                audio_path = audio_file.name
            
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
                output_path = output_file.name
            
            # Run Rhubarb CLI
            result = subprocess.run(
                [
                    self.rhubarb_path,
                    "-f", "json",                    # Output format
                    "-o", output_path,               # Output file
                    "--machineReadable",             # Suppress progress output
                    audio_path                       # Input audio
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Rhubarb failed: {result.stderr}")
                return None
            
            # Parse output
            with open(output_path, "r") as f:
                data = json.load(f)
            
            visemes = []
            for cue in data.get("mouthCues", []):
                visemes.append(Viseme(
                    time=cue["start"],
                    shape=cue["value"]
                ))
            
            duration = data.get("metadata", {}).get("duration", 0.0)
            
            # Cleanup temp files
            Path(audio_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
            
            logger.info(f"Extracted {len(visemes)} visemes from {duration:.2f}s audio")
            return LipSyncResult(visemes=visemes, duration=duration)
            
        except Exception as e:
            logger.error(f"Viseme extraction failed: {e}")
            return None
    
    def visemes_to_dict(self, result: LipSyncResult) -> dict:
        """Convert LipSyncResult to JSON-serializable dict."""
        return {
            "duration": result.duration,
            "visemes": [
                {"time": v.time, "shape": v.shape}
                for v in result.visemes
            ]
        }

# Internal singleton instance
_INSTANCE: Optional["RhubarbService"] = None

def get_rhubarb_service() -> RhubarbService:
    """Get or create the singleton instance of RhubarbService."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RhubarbService()
    return _INSTANCE
