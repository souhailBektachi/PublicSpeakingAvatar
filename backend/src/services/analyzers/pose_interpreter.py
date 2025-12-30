import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

UPPER_BODY_LANDMARKS = [11, 12, 13, 14, 15, 16]
THORAX_LANDMARK = 12


class PoseFeedbackEngine:
    """Interprets pose + face landmarks and produces metrics + textual feedback.

    Assumes MediaPipe-like indices for body (pose) and face landmarks.
    """

    def __init__(self):
        self.prev_landmarks: Optional[List[Tuple[float, float, float]]] = None
        self.z_positions: List[float] = []
        self.stage_centers: List[Tuple[float, float]] = []
        self.gesture_vectors: List[List[Tuple[float, float, float]]] = []
        self.prev_face_landmarks: Optional[List[Dict[str, float]]] = None
        # cache shoulder width for normalization-derived heuristics
        self._last_shoulder_width: float = 1.0
        
        # Heuristics State
        self.script_reading_frames = 0
        self.looking_away_frames = 0
        self.calibration = {"neutral_neck_height": None}

    # ---------- NORMALIZATION ----------
    def normalize_by_thorax(self, landmarks: List[Dict[str, float]]) -> List[Tuple[float, float, float]]:
        if not landmarks or len(landmarks) <= THORAX_LANDMARK:
            return [(0.0, 0.0, 0.0) for _ in range(max(len(landmarks), 33))]

        thorax = landmarks[THORAX_LANDMARK]
        left = landmarks[11] if len(landmarks) > 11 else thorax
        right = landmarks[12] if len(landmarks) > 12 else thorax

        shoulder_width = abs(left.get("x", 0.0) - right.get("x", 0.0)) or 1.0
        self._last_shoulder_width = shoulder_width

        norm: List[Tuple[float, float, float]] = []
        for lm in landmarks:
            x = (lm.get("x", 0.0) - thorax.get("x", 0.0)) / shoulder_width
            y = (lm.get("y", 0.0) - thorax.get("y", 0.0)) / shoulder_width
            z = (lm.get("z", 0.0) - thorax.get("z", 0.0)) / shoulder_width
            norm.append((x, y, z))
        return norm

    # ---------- METRICS ----------
    def gesture_energy(self, curr: List[Tuple[float, float, float]]) -> float:
        if not self.prev_landmarks:
            self.prev_landmarks = curr
            return 0.0

        energy = 0.0
        for i in UPPER_BODY_LANDMARKS:
            if i < len(curr) and i < len(self.prev_landmarks):
                dx = curr[i][0] - self.prev_landmarks[i][0]
                dy = curr[i][1] - self.prev_landmarks[i][1]
                energy += float(np.sqrt(dx * dx + dy * dy))

        self.prev_landmarks = curr
        return energy / max(1, len(UPPER_BODY_LANDMARKS))

    def gesture_diversity(self, curr: List[Tuple[float, float, float]]) -> float:
        if not self.gesture_vectors:
            self.gesture_vectors.append(curr)
            return 0.0

        ref = self.gesture_vectors[0]
        dists: List[float] = []

        for i in UPPER_BODY_LANDMARKS:
            if i < len(curr) and i < len(ref):
                v1 = np.array(curr[i][:2], dtype=np.float32)
                v2 = np.array(ref[i][:2], dtype=np.float32)
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                if denom > 0:
                    dists.append(1.0 - float(np.dot(v1, v2) / denom))

        return float(np.std(dists)) if dists else 0.0

    def z_volatility(self, landmarks: List[Dict[str, float]]) -> float:
        # Use landmark 0 (nose) depth if present, else average z
        z_val: float
        if landmarks and 0 < len(landmarks):
            z_val = float(landmarks[0].get("z", 0.0))
        else:
            z_vals = [lm.get("z", 0.0) for lm in landmarks]
            z_val = float(np.mean(z_vals)) if z_vals else 0.0

        self.z_positions.append(z_val)
        self.z_positions = self.z_positions[-30:]
        return float(np.std(self.z_positions)) if len(self.z_positions) > 1 else 0.0

    def dispersion(self, landmarks: List[Dict[str, float]]) -> float:
        if not landmarks:
            return 0.0
        xs = [lm.get("x", 0.0) for lm in landmarks]
        ys = [lm.get("y", 0.0) for lm in landmarks]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        self.stage_centers.append((cx, cy))
        self.stage_centers = self.stage_centers[-60:]

        if len(self.stage_centers) > 1:
            sx, sy = zip(*self.stage_centers)
            return (float(np.std(sx)) + float(np.std(sy))) / 2.0
        return 0.0

    # ---------- FACE ----------
    def valence_arousal(self, face: Optional[List[Dict[str, float]]]) -> Tuple[float, float]:
        if not face or len(face) < 292:
            return 0.0, 0.0

        mouth_l = face[61]
        mouth_r = face[291]
        nose = face[1]
        valence = abs(float(mouth_r.get("x", 0.0)) - float(mouth_l.get("x", 0.0))) \
            + (1.0 - abs(float(nose.get("y", 0.0)) - float(mouth_l.get("y", 0.0))))

        arousal = 0.0
        if self.prev_face_landmarks and len(self.prev_face_landmarks) > 105:
            arousal = abs(float(face[105].get("x", 0.0)) - float(self.prev_face_landmarks[105].get("x", 0.0))) * 10.0

        self.prev_face_landmarks = face
        return valence, arousal

    # ---------- FEEDBACK EXPORT ----------
    def generate_feedback(self, metrics: Tuple[float, float, float, float, float, float]) -> List[str]:
        energy, diversity, dispersion, z, valence, arousal = metrics
        feedback: List[str] = []

        feedback.append(
            "EXCELLENT gestures" if energy > 0.15 else
            "GOOD gestures" if energy > 0.1 else
            "USE MORE GESTURES"
        )

        feedback.append(
            "VARIED gestures" if diversity > 0.3 else
            "REPETITIVE gestures"
        )

        feedback.append(
            "GOOD stage movement" if dispersion > 0.1 else
            "STATIC position"
        )

        feedback.append(
            "DYNAMIC depth" if z > 0.02 else
            "FLAT depth"
        )

        feedback.append(
            "POSITIVE expression" if valence > 0.5 else
            "LOW expression"
        )

        feedback.append(
            "ENERGETIC face" if arousal > 0.3 else
            "LOW facial energy"
        )

        return feedback

    # ---------- ARMS & POSTURE ----------
    @staticmethod
    def _angle(a: Dict[str, float], b: Dict[str, float], c: Dict[str, float]) -> float:
        """Angle at b (degrees) formed by points a-b-c."""
        ax, ay = a.get("x", 0.0), a.get("y", 0.0)
        bx, by = b.get("x", 0.0), b.get("y", 0.0)
        cx, cy = c.get("x", 0.0), c.get("y", 0.0)
        v1 = np.array([ax - bx, ay - by], dtype=np.float32)
        v2 = np.array([cx - bx, cy - by], dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def arms_metrics(self, pose: List[Dict[str, float]]) -> Dict[str, float | bool]:
        """Compute arm-related metrics and simple posture flags."""
        res: Dict[str, float | bool] = {}
        if len(pose) < 17:  # need shoulders, elbows, wrists
            return {
                "left_elbow_deg": 0.0,
                "right_elbow_deg": 0.0,
                "left_arm_raised": False,
                "right_arm_raised": False,
                "hands_distance": 0.0,
                "hands_together": False,
                "arms_crossed": False,
            }

        ls, rs = pose[11], pose[12]
        le, re = pose[13], pose[14]
        lw, rw = pose[15], pose[16]

        left_elbow_deg = self._angle(ls, le, lw)
        right_elbow_deg = self._angle(rs, re, rw)

        # In image coords, smaller y is higher (arm raised if wrist above shoulder)
        left_arm_raised = lw.get("y", 0.0) < ls.get("y", 0.0) - 0.02
        right_arm_raised = rw.get("y", 0.0) < rs.get("y", 0.0) - 0.02

        # Hands distance normalized by shoulder width
        shoulder_w = self._last_shoulder_width or max(abs(ls.get("x", 0.0) - rs.get("x", 0.0)), 1e-6)
        hands_distance = float(np.linalg.norm(
            np.array([lw.get("x", 0.0) - rw.get("x", 0.0), lw.get("y", 0.0) - rw.get("y", 0.0)], dtype=np.float32)
        )) / shoulder_w

        hands_together = hands_distance < 0.3

        # Arms crossed: wrists near midline and left wrist right of thorax or vice-versa
        thorax_x = pose[THORAX_LANDMARK].get("x", (ls.get("x", 0.0) + rs.get("x", 0.0)) / 2.0)
        arms_crossed = (lw.get("x", 0.0) > thorax_x and rw.get("x", 0.0) < thorax_x) and hands_together

        res.update({
            "left_elbow_deg": left_elbow_deg,
            "right_elbow_deg": right_elbow_deg,
            "left_arm_raised": left_arm_raised,
            "right_arm_raised": right_arm_raised,
            "hands_distance": hands_distance,
            "hands_together": hands_together,
            "arms_crossed": arms_crossed,
        })
        return res

    def arms_feedback(self, arms: Dict[str, float | bool]) -> List[str]:
        msgs: List[str] = []
        if arms.get("left_arm_raised") and arms.get("right_arm_raised"):
            msgs.append("Both arms raised")
        elif arms.get("left_arm_raised"):
            msgs.append("Left arm raised")
        elif arms.get("right_arm_raised"):
            msgs.append("Right arm raised")

        le = float(arms.get("left_elbow_deg", 0.0))
        re = float(arms.get("right_elbow_deg", 0.0))
        if le > 160 and re > 160:
            msgs.append("Arms extended")
        elif le < 70 or re < 70:
            msgs.append("Strong bend at elbow")

        if arms.get("arms_crossed"):
            msgs.append("Closed posture (arms crossed)")
        elif arms.get("hands_together"):
            msgs.append("Hands together")
        else:
            msgs.append("Open posture")
        return msgs

    # ---------- ADVANCED HEURISTICS ----------
    def analyze_eye_contact(self, face: List[Dict[str, float]]) -> Dict[str, float | str | bool]:
        """
        Analyze gaze deviation to detect eye contact or looking away.
        Uses Iris Deviation from Eye Center heuristics.
        """
        if not face or len(face) < 474:
            return {"deviation": 0.0, "status": "unknown"}

        # Indices
        LEFT_IRIS = 468
        RIGHT_IRIS = 473
        LEFT_EYE_CORNERS = (33, 133)
        RIGHT_EYE_CORNERS = (362, 263)

        def get_pt(idx): return np.array([face[idx].get("x",0), face[idx].get("y",0)])

        # Left Eye Calculation
        l_iris = get_pt(LEFT_IRIS)
        l_inner, l_outer = get_pt(LEFT_EYE_CORNERS[0]), get_pt(LEFT_EYE_CORNERS[1])
        l_width = np.linalg.norm(l_outer - l_inner)
        l_center = (l_inner + l_outer) / 2.0
        # Deviation normalized by eye width
        l_dev = np.linalg.norm(l_iris - l_center) / (l_width + 1e-6)

        # Right Eye Calculation
        r_iris = get_pt(RIGHT_IRIS)
        r_inner, r_outer = get_pt(RIGHT_EYE_CORNERS[0]), get_pt(RIGHT_EYE_CORNERS[1])
        r_width = np.linalg.norm(r_outer - r_inner)
        r_center = (r_inner + r_outer) / 2.0
        r_dev = np.linalg.norm(r_iris - r_center) / (r_width + 1e-6)

        avg_dev = (l_dev + r_dev) / 2.0

        # State Tracking
        if avg_dev > 0.15: # Threshold for looking away
            self.looking_away_frames += 1
        else:
            self.looking_away_frames = 0
            
        status = "camera"
        if self.looking_away_frames > 30: # ~1-2 seconds
            status = "distracted"
        
        return {
            "deviation": float(avg_dev),
            "status": status,
            "looking_at_camera": avg_dev < 0.15
        }

    def analyze_script_reading(self, pose: List[Dict[str, float]]) -> Dict[str, float | bool]:
        """
        Detects if user is reading from a script by checking sustained head-down posture.
        Uses Nose vs Ears vertical alignment.
        """
        if not pose or len(pose) < 9: # Need nose(0) and ears(7,8)
            return {"is_reading": False}

        nose_y = pose[0].get("y", 0.0)
        left_ear_y = pose[7].get("y", 0.0)
        right_ear_y = pose[8].get("y", 0.0)
        avg_ear_y = (left_ear_y + right_ear_y) / 2.0

        # In MediaPipe Image coords, Y increases downwards.
        # If looking down, Nose Y increases (moves down) relative to Ears.
        # If Nose is significantly "below" ears (y > ear_y), head is tilted down.
        # We need a robust delta.
        
        # Calibration: Capture neutral on first few frames? 
        # Heuristic: Nose usually slightly below ears in neutral 2D projection.
        # Let's use a change detection or absolute threshold.
        # Assume if nose is > 0.05 units below ears implies tilt.
        
        neck_tilt_scan = nose_y - avg_ear_y # Positive means nose is below ears
        
        # Threshold: > 0.08 seems like a distinct nod/look down in normalized coords
        is_head_down = neck_tilt_scan > 0.06 

        if is_head_down:
            self.script_reading_frames += 1
        else:
            self.script_reading_frames = 0

        # Trigger if sustained for ~2-3 seconds (assuming ~15-30fps, say 45 frames)
        is_reading = self.script_reading_frames > 45

        return {
            "head_tilt_val": float(neck_tilt_scan),
            "is_head_down": is_head_down,
            "is_reading": is_reading
        }

    def analyze_micro_gestures(self, pose: List[Dict[str, float]], face: List[Dict[str, float]]) -> Dict[str, bool]:
        """
        Detect hand-to-face signals (nervousness/thinking).
        """
        if not pose or not face:
            return {"hand_near_face": False}

        # Face Height
        forehead = face[10]
        chin = face[152]
        face_height = np.linalg.norm(np.array([forehead.get("x",0), forehead.get("y",0)]) - 
                                     np.array([chin.get("x",0), chin.get("y",0)]))
        
        # Face Center (using Nose)
        nose = pose[0]
        nose_pt = np.array([nose.get("x",0), nose.get("y",0)])
        
        # Hands
        left_wrist = np.array([pose[15].get("x",0), pose[15].get("y",0)])
        right_wrist = np.array([pose[16].get("x",0), pose[16].get("y",0)])

        threshold = face_height * 1.5

        l_dist = np.linalg.norm(left_wrist - nose_pt)
        r_dist = np.linalg.norm(right_wrist - nose_pt)

        is_near = (l_dist < threshold) or (r_dist < threshold)
        
        return {"hand_near_face": bool(is_near)}
    def process_chunk(self, chunk: Dict) -> Dict:
        pose_data: Dict = chunk.get("pose_data", {})
        timestamp: float = float(chunk.get("timestamp", 0.0))
        session_id: Optional[str] = chunk.get("session_id")

        # Extract pose landmarks (use first frame if nested list)
        pose_landmarks: List[Dict[str, float]] = []
        raw_pose = pose_data.get("poseLandmarks")
        if isinstance(raw_pose, list):
            if raw_pose and isinstance(raw_pose[0], list):
                pose_landmarks = raw_pose[0]
            else:
                pose_landmarks = raw_pose  # already flat

        # Extract face landmarks similarly
        face_landmarks: Optional[List[Dict[str, float]]] = None
        raw_face = pose_data.get("faceLandmarks")
        if isinstance(raw_face, list):
            if raw_face and isinstance(raw_face[0], list):
                face_landmarks = raw_face[0]
            else:
                face_landmarks = raw_face

        norm = self.normalize_by_thorax(pose_landmarks)

        metrics = (
            self.gesture_energy(norm),
            self.gesture_diversity(norm),
            self.dispersion(pose_landmarks),
            self.z_volatility(pose_landmarks),
            *self.valence_arousal(face_landmarks)
        )

        arms = self.arms_metrics(pose_landmarks)
        arms_fb = self.arms_feedback(arms)

        # Advanced Heuristics
        eye_metrics = self.analyze_eye_contact(face_landmarks)
        script_reading = self.analyze_script_reading(pose_landmarks)
        micro_gesture = self.analyze_micro_gestures(pose_landmarks, face_landmarks)

        # --- LLM-ready summary (concise) ---
        def level(v: float, lo: float, hi: float) -> str:
            return "low" if v < lo else "medium" if v < hi else "high"

        gestures_level = level(metrics[0], 0.08, 0.15)  # energy
        expression_level = level(metrics[4], 0.3, 0.6)   # valence
        face_energy_level = level(metrics[5], 0.2, 0.4)  # arousal
        movement_level = "static" if metrics[2] < 0.08 else "moving"

        # Horizontal position from normalized center
        norm_cx = float(np.mean([p[0] for p in norm])) if norm else 0.0
        if norm_cx < -0.2:
            horiz = "left"
        elif norm_cx > 0.2:
            horiz = "right"
        else:
            horiz = "center"

        # Depth using recent z mean (negative forward in many models; keep simple)
        z_mean = float(np.mean(self.z_positions)) if self.z_positions else 0.0
        depth = "forward" if z_mean < -0.01 else "middle" if abs(z_mean) <= 0.03 else "back"

        elbows_state = (
            "straight" if arms["left_elbow_deg"] > 150 and arms["right_elbow_deg"] > 150
            else "bent" if arms["left_elbow_deg"] < 90 or arms["right_elbow_deg"] < 90
            else "neutral"
        )

        posture_open = not arms.get("arms_crossed", False)
        hands_state = "together" if arms.get("hands_together", False) else "apart"
        
        # New Cognitive / Heuristic States
        eye_status = str(eye_metrics.get("status", "unknown"))
        reading_status = bool(script_reading.get("is_reading", False))
        nervous_gesture = bool(micro_gesture.get("hand_near_face", False))

        summary = {
            "posture": {
                "open": posture_open,
                "arms": {
                    "left_raised": bool(arms.get("left_arm_raised", False)),
                    "right_raised": bool(arms.get("right_arm_raised", False)),
                    "elbows": elbows_state,
                    "hands": hands_state,
                    "crossed": bool(arms.get("arms_crossed", False)),
                    "hand_near_face": nervous_gesture
                },
                "head": {
                   "is_reading_script": reading_status,
                   "looking_at": eye_status
                }
            },
            "behavior": {
                "gestures": gestures_level,
                "facial_expression": expression_level,
                "facial_energy": face_energy_level,
                "movement": movement_level,
            },
            "position": {
                "horizontal": horiz,
                "depth": depth,
            },
        }

        # Append advanced feedback
        feedbacks = self.generate_feedback(metrics) + arms_fb
        if reading_status:
           feedbacks.append("Potential script reading detected")
        if nervous_gesture:
           feedbacks.append("Hand touching face detected")
        if eye_status == "distracted":
           feedbacks.append("Low eye contact")

        return {
            "session_id": session_id,
            "timestamp": timestamp,
            "metrics": {
                "energy": metrics[0],
                "diversity": metrics[1],
                "dispersion": metrics[2],
                "z_volatility": metrics[3],
                "valence": metrics[4],
                "arousal": metrics[5],
                "left_elbow_deg": arms["left_elbow_deg"],
                "right_elbow_deg": arms["right_elbow_deg"],
                "hands_distance": arms["hands_distance"],
                "left_arm_raised": arms["left_arm_raised"],
                "right_arm_raised": arms["right_arm_raised"],
                "hands_together": arms["hands_together"],
                "arms_crossed": arms["arms_crossed"],
                "gaze_deviation": eye_metrics.get("deviation", 0.0),
                "head_tilt": script_reading.get("head_tilt_val", 0.0)
            },
            "feedback": feedbacks,
            "summary": summary,
        }