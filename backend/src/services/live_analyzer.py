import logging
import base64
import random
from pathlib import Path
from typing import Optional, Dict
from src.schemas.audio_metrics import AudioFeatures

logger = logging.getLogger(__name__)

class LiveAnalyzer:
    def __init__(self):
        self.info_counter = 0
        self.info_threshold = 3
        self.warning_counter = 0
        self.warning_threshold = 3
        self.audio_base_path = Path(__file__).parent.parent.parent / "assets" / "audio" / "feedback"
        
    def analyze(self, metrics: AudioFeatures) -> Optional[Dict[str, str]]:
        if not metrics.feedback:
            return None

        warning_feedback = self._check_warnings(metrics)
        if warning_feedback:
            return warning_feedback

        info_feedback = self._check_info(metrics)
        if info_feedback:
            return info_feedback

        return None

    def _check_warnings(self, metrics: AudioFeatures) -> Optional[Dict[str, str]]:
        warnings = []
        
        if metrics.feedback.volume.severity == "warning":
            warnings.append(("volume", metrics.feedback.volume))

        if metrics.feedback.hesitation.severity == "warning":
            warnings.append(("hesitation", metrics.feedback.hesitation))

        if metrics.feedback.pitch_dynamics.severity == "warning":
            warnings.append(("pitch_dynamics", metrics.feedback.pitch_dynamics))

        if not warnings:
            return None

        self.warning_counter += 1
        if self.warning_counter >= self.warning_threshold:
            self.warning_counter = 0
            category, item = warnings[0]
            return self._get_audio_response(category, item.status, item.message, item.message_id)

        return None

    def _check_info(self, metrics: AudioFeatures) -> Optional[Dict[str, str]]:
        info_items = []
        
        if metrics.feedback.fluency.severity == "info":
            info_items.append(("fluency", metrics.feedback.fluency))
        
        if metrics.feedback.pitch_range.severity == "info":
            info_items.append(("pitch_range", metrics.feedback.pitch_range))

        if not info_items:
            return None

        self.info_counter += 1
        if self.info_counter >= self.info_threshold:
            self.info_counter = 0
            category, item = random.choice(info_items)
            return self._get_audio_response(category, item.status, item.message, item.message_id)
            
        return None

    def _get_audio_response(self, category: str, status: str, message: str, message_id: int) -> Optional[Dict[str, str]]:
        if not self.audio_base_path.exists():
            return None
            
        safe_category = category.lower()
        # Try with underscores first (e.g. Fluent_Fast)
        safe_status = status.replace("/", "_").replace(" ", "_")
        
        filename = f"{safe_category}_{safe_status}_{message_id}.mp3"
        audio_path = self.audio_base_path / filename
        
        # Fallback: Try removing spaces (e.g. TooSoft)
        if not audio_path.exists():
            safe_status_nospace = status.replace("/", "_").replace(" ", "")
            filename_nospace = f"{safe_category}_{safe_status_nospace}_{message_id}.mp3"
            if (self.audio_base_path / filename_nospace).exists():
                audio_path = self.audio_base_path / filename_nospace
                safe_status = safe_status_nospace # Update for pattern matching below
        
        if not audio_path.exists():
            try:
                pattern = f"{safe_category}_{safe_status}_*.mp3"
                files = list(self.audio_base_path.glob(pattern))
                if files:
                    audio_path = random.choice(files)
            except Exception:
                pass
            
        if not audio_path.exists():
            logger.warning(f"No audio file found for {category}/{status}/{message_id} at {audio_path}")
            return None

        try:
            with open(audio_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")
                
            return {
                "type": "audio_feedback",
                "audio": audio_data
            }
        except Exception as e:
            logger.error(f"Error reading audio file {audio_path}: {e}")
            return None

