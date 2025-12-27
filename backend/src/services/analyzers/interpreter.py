import numpy as np
from typing import List
from src.schemas.audio_metrics import AudioFeatures, ProsodyFeedback, FeedbackItem, VolumeFeedbackItem

class ProsodyInterpreter:
    def __init__(self):
        self.male_pitch_range = (85, 180)
        self.female_pitch_range = (165, 255)
        self.reference_rms = 0.1
        
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
            return FeedbackItem(
                status="Monotone",
                message="Your delivery is flat. Try varying your intonation to emphasize key words.",
                severity="warning"
            )
        elif pitch_std <= 40:
            return FeedbackItem(
                status="Typical",
                message="Good natural pitch variation.",
                severity="success"
            )
        else:
            return FeedbackItem(
                status="Expressive",
                message="Highly expressive and engaging tone.",
                severity="success"
            )
    
    def _analyze_pitch_range(self, pitch_mean: float) -> FeedbackItem:
        if pitch_mean == 0:
            return FeedbackItem(
                status="Unknown",
                message="No pitch data available.",
                severity="info"
            )
        
        in_male_range = self.male_pitch_range[0] <= pitch_mean <= self.male_pitch_range[1]
        in_female_range = self.female_pitch_range[0] <= pitch_mean <= self.female_pitch_range[1]
        
        if in_male_range or in_female_range:
            return FeedbackItem(
                status="Normal",
                message=f"Pitch at {pitch_mean:.1f} Hz is within healthy vocal range.",
                severity="success"
            )
        elif pitch_mean < self.male_pitch_range[0]:
            return FeedbackItem(
                status="Vocal Fry",
                message="Pitch is unusually low. This may strain your voice or sound disengaged.",
                severity="warning"
            )
        elif pitch_mean > self.female_pitch_range[1]:
            return FeedbackItem(
                status="Vocal Strain",
                message="Pitch is unusually high. Relax your throat to avoid strain.",
                severity="warning"
            )
        else:
            return FeedbackItem(
                status="Normal",
                message=f"Pitch at {pitch_mean:.1f} Hz is acceptable.",
                severity="info"
            )
    
    def _analyze_fluency(self, voiced_prob: float) -> FeedbackItem:
        if voiced_prob < 0.5:
            return FeedbackItem(
                status="Pausing/Slow",
                message="High pause frequency detected. Consider speaking more continuously.",
                severity="info"
            )
        elif voiced_prob <= 0.7:
            return FeedbackItem(
                status="Typical",
                message="Balanced speech with natural pacing.",
                severity="success"
            )
        else:
            return FeedbackItem(
                status="Fluent/Fast",
                message="Continuous delivery. Ensure your audience can keep up.",
                severity="info"
            )
    
    def _analyze_hesitation(self, hesitation_rate: float, volume: float) -> FeedbackItem:
        silence_threshold = 0.01
        
        if hesitation_rate > 0.1 and volume > silence_threshold:
            return FeedbackItem(
                status="Detected",
                message="Filled pause detected (Um/Uh). Try pausing silently instead.",
                severity="warning"
            )
        elif hesitation_rate > 0.0:
            return FeedbackItem(
                status="Minor",
                message="Slight hesitation detected. Monitor your fluency.",
                severity="info"
            )
        else:
            return FeedbackItem(
                status="Clear",
                message="No hesitations detected. Great fluency!",
                severity="success"
            )
    
    def _analyze_volume(self, rms: float) -> VolumeFeedbackItem:
        if rms < 1e-6:
            db_fs = -120.0
        else:
            db_fs = 20 * np.log10(rms / self.reference_rms)
        
        if db_fs < -40:
            return VolumeFeedbackItem(
                status="Too Soft",
                message=f"Volume is too low ({db_fs:.1f} dBFS). Speak louder for better projection.",
                severity="warning",
                db_fs=db_fs
            )
        elif db_fs > -3:
            return VolumeFeedbackItem(
                status="Clipping",
                message=f"Volume is too high ({db_fs:.1f} dBFS). Reduce volume to avoid distortion.",
                severity="warning",
                db_fs=db_fs
            )
        elif -20 <= db_fs <= -10:
            return VolumeFeedbackItem(
                status="Optimal",
                message=f"Excellent volume level ({db_fs:.1f} dBFS).",
                severity="success",
                db_fs=db_fs
            )
        else:
            return VolumeFeedbackItem(
                status="Acceptable",
                message=f"Volume is acceptable ({db_fs:.1f} dBFS).",
                severity="info",
                db_fs=db_fs
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