import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from src.schemas.audio_metrics import AudioFeatures, ProsodyFeedback, FeedbackItem, VolumeFeedbackItem

class ProsodyInterpreter:
    def __init__(self):
        self.male_pitch_range = (85, 180)
        self.female_pitch_range = (165, 255)
        self.reference_rms = 0.1
        self.messages = self._load_messages()
        
    def _load_messages(self) -> Dict:
        try:
            path = Path(__file__).parent.parent.parent / "core" / "feedback_messages.json"
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading feedback messages: {e}")
            return {}

    def _get_message(self, category: str, status: str, default: str) -> Tuple[str, int]:
        category_msgs = self.messages.get(category, {})
        status_msgs = category_msgs.get(status, [])
        if status_msgs:
            idx = random.randrange(len(status_msgs))
            return status_msgs[idx], idx
        return default, 0

    def interpret(self, features: AudioFeatures) -> ProsodyFeedback:
        pitch_dynamics = self._analyze_pitch_dynamics(features.pitch_std)
        pitch_range = self._analyze_pitch_range(features.pitch_mean)
        fluency = self._analyze_fluency(features.voiced_prob)
        hesitation = self._analyze_hesitation(features.hesitation_rate, features.volume)
        volume = self._analyze_volume(features.volume)
        
        temp_feedback = {
            "pitch_dynamics": pitch_dynamics,
            "pitch_range": pitch_range,
            "fluency": fluency,
            "hesitation": hesitation,
            "volume": volume
        }
        
        suggestions = self._generate_suggestions(temp_feedback)
        overall_score = self._calculate_score(temp_feedback)
        
        return ProsodyFeedback(
            pitch_dynamics=pitch_dynamics,
            pitch_range=pitch_range,
            fluency=fluency,
            hesitation=hesitation,
            volume=volume,
            overall_score=overall_score,
            suggestions=suggestions
        )
    
    def _analyze_pitch_dynamics(self, pitch_std: float) -> FeedbackItem:
        if pitch_std < 10:
            status = "Monotone"
            msg, msg_id = self._get_message("pitch_dynamics", status, "Your delivery is a bit flat.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="warning",
                message_id=msg_id
            )
        elif pitch_std <= 40:
            status = "Typical"
            msg, msg_id = self._get_message("pitch_dynamics", status, "Good natural pitch variation.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="success",
                message_id=msg_id
            )
        else:
            status = "Expressive"
            msg, msg_id = self._get_message("pitch_dynamics", status, "Highly expressive and engaging tone.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="success",
                message_id=msg_id
            )
    
    def _analyze_pitch_range(self, pitch_mean: float) -> FeedbackItem:
        if pitch_mean == 0:
            status = "Unknown"
            msg, msg_id = self._get_message("pitch_range", status, "No pitch data available.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="info",
                message_id=msg_id
            )
        
        in_male_range = self.male_pitch_range[0] <= pitch_mean <= self.male_pitch_range[1]
        in_female_range = self.female_pitch_range[0] <= pitch_mean <= self.female_pitch_range[1]
        
        if in_male_range or in_female_range:
            status = "Normal"
            msg, msg_id = self._get_message("pitch_range", status, "Your pitch is within a healthy vocal range.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="success",
                message_id=msg_id
            )
        elif pitch_mean < self.male_pitch_range[0]:
            status = "Vocal Fry"
            msg, msg_id = self._get_message("pitch_range", status, "Your pitch is unusually low.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="warning",
                message_id=msg_id
            )
        elif pitch_mean > self.female_pitch_range[1]:
            status = "Vocal Strain"
            msg, msg_id = self._get_message("pitch_range", status, "Your pitch is unusually high.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="warning",
                message_id=msg_id
            )
        else:
            status = "Normal"
            msg, msg_id = self._get_message("pitch_range", status, "Your pitch is acceptable.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="info",
                message_id=msg_id
            )
    
    def _analyze_fluency(self, voiced_prob: float) -> FeedbackItem:
        if voiced_prob < 0.5:
            status = "Pausing/Slow"
            msg, msg_id = self._get_message("fluency", status, "High pause frequency detected.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="info",
                message_id=msg_id
            )
        elif voiced_prob <= 0.7:
            status = "Typical"
            msg, msg_id = self._get_message("fluency", status, "Balanced speech with natural pacing.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="success",
                message_id=msg_id
            )
        else:
            status = "Fluent/Fast"
            msg, msg_id = self._get_message("fluency", status, "Continuous delivery.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="info",
                message_id=msg_id
            )
    
    def _analyze_hesitation(self, hesitation_rate: float, volume: float) -> FeedbackItem:
        silence_threshold = 0.01
        
        if hesitation_rate > 0.1 and volume > silence_threshold:
            status = "Detected"
            msg, msg_id = self._get_message("hesitation", status, "Filled pause detected (Um/Uh).")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="warning",
                message_id=msg_id
            )
        elif hesitation_rate > 0.0:
            status = "Minor"
            msg, msg_id = self._get_message("hesitation", status, "Slight hesitation detected.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="info",
                message_id=msg_id
            )
        else:
            status = "Clear"
            msg, msg_id = self._get_message("hesitation", status, "No hesitations detected.")
            return FeedbackItem(
                status=status,
                message=msg,
                severity="success",
                message_id=msg_id
            )
    
    def _analyze_volume(self, rms: float) -> VolumeFeedbackItem:
        if rms < 1e-6:
            db_fs = -120.0
        else:
            db_fs = 20 * np.log10(rms / self.reference_rms)
        
        if db_fs < -40:
            status = "Too Soft"
            msg, msg_id = self._get_message("volume", status, "Your volume is a bit low.")
            return VolumeFeedbackItem(
                status=status,
                message=msg,
                severity="warning",
                db_fs=db_fs,
                message_id=msg_id
            )
        elif db_fs > -3:
            status = "Clipping"
            msg, msg_id = self._get_message("volume", status, "Your volume is too high.")
            return VolumeFeedbackItem(
                status=status,
                message=msg,
                severity="warning",
                db_fs=db_fs,
                message_id=msg_id
            )
        elif -20 <= db_fs <= -10:
            status = "Optimal"
            msg, msg_id = self._get_message("volume", status, "Excellent volume level.")
            return VolumeFeedbackItem(
                status=status,
                message=msg,
                severity="success",
                db_fs=db_fs,
                message_id=msg_id
            )
        else:
            status = "Acceptable"
            msg, msg_id = self._get_message("volume", status, "Your volume is at a good level.")
            return VolumeFeedbackItem(
                status=status,
                message=msg,
                severity="info",
                db_fs=db_fs,
                message_id=msg_id
            )
    
    def _generate_suggestions(self, feedback: dict) -> List[str]:
        suggestions = []
        
        if feedback["pitch_dynamics"].severity == "warning":
            suggestions.append("Practice emphasizing keywords with pitch changes")
        
        if feedback["pitch_range"].severity == "warning":
            suggestions.append("Adjust your vocal pitch to a more comfortable range")
        
        if feedback["hesitation"].status == "Detected":
            suggestions.append("Replace 'um' and 'uh' with brief silent pauses")
        
        if feedback["volume"].severity == "warning":
            if feedback["volume"].status == "Too Soft":
                suggestions.append("Increase volume by projecting from your diaphragm")
            else:
                suggestions.append("Lower your volume to prevent audio clipping")
        
        if feedback["fluency"].status == "Fluent/Fast":
            suggestions.append("Consider adding strategic pauses for emphasis")
        
        if not suggestions:
            suggestions.append("Keep up the great work!")
        
        return suggestions
    
    def _calculate_score(self, feedback: dict) -> float:
        score = 100.0
        
        if feedback["pitch_dynamics"].severity == "warning":
            score -= 15
        
        if feedback["pitch_range"].severity == "warning":
            score -= 10
        
        if feedback["hesitation"].status == "Detected":
            score -= 20
        elif feedback["hesitation"].status == "Minor":
            score -= 5
        
        if feedback["volume"].severity == "warning":
            score -= 15
        
        if feedback["fluency"].status == "Pausing/Slow":
            score -= 5
        
        return max(0.0, score)