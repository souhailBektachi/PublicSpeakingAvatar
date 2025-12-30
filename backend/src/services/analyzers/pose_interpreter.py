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

    # ---------- MAIN ENTRY ----------
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

        summary = {
            "posture": {
                "open": posture_open,
                "arms": {
                    "left_raised": bool(arms.get("left_arm_raised", False)),
                    "right_raised": bool(arms.get("right_arm_raised", False)),
                    "elbows": elbows_state,
                    "hands": hands_state,
                    "crossed": bool(arms.get("arms_crossed", False)),
                },
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
            },
            "feedback": self.generate_feedback(metrics) + arms_fb,
            "summary": summary,
        }