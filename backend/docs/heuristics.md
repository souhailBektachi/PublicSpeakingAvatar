
### **1. Speaking Rate (Pace)**
*   **Definition:** The speed of speech delivery measured in Words Per Minute (WPM).
*   **Normalization:** None required (Time and Word Count are absolute).
*   **How to Calculate:**
    1.  Use **Whisper** to get the transcript with start/end timestamps.
    2.  `Word_Count = len(transcript.split())`
    3.  `Duration_Minutes = (end_time - start_time) / 60`
    4.  `WPM = Word_Count / Duration_Minutes`
*   **Thresholds:**
    *   **< 110 WPM:** Too Slow (Low Energy/Boring).
    *   **130 – 160 WPM:** **Ideal Range** (Conversational/Persuasive).
    *   **> 160 WPM:** Too Fast (Nervous/Rushing).
*   **Implications:** Fast speech reduces comprehension. Slow speech reduces engagement.
*   **Source:** *SpeechMirror* (Pace analysis) / Standard Linguistics.

---

### **2. Gesture Frequency**
*   **Definition:** The number of distinct, meaningful hand movements made per minute.
*   **Normalization:** **Shoulder Width**. Determine the "Movement Threshold" based on the distance between MediaPipe Landmarks 11 (Left Shoulder) and 12 (Right Shoulder).
*   **How to Calculate:**
    1.  Track wrists (Landmarks 15, 16).
    2.  Calculate velocity between frames: $v = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$.
    3.  If $v > (0.1 \times \text{ShoulderWidth})$, count it as a "Gesture Event."
    4.  Sum events over the session duration.
*   **Thresholds:**
    *   **< 7 gestures/min:** Low Energy (Passive/Reserved).
    *   **8 – 14 gestures/min:** Moderate (Standard).
    *   **> 15 gestures/min:** **High Energy (Persuasive/Dynamic).**
*   **Implications:** High-frequency gesturing is correlated with higher persuasion and clarity in political/public speaking.
*   **Source:** *Biomechanics of Public Speaking (Ding, 2024)* (Table 6).

---

### **3. Posture Openness**
*   **Definition:** Whether the speaker’s body language is "Open" (welcoming) or "Closed" (defensive).
*   **Normalization:** **Torso Box**. Defined by the rectangle formed by Shoulders (11, 12) and Hips (23, 24).
*   **How to Calculate:**
    1.  Get X-coordinates of Wrists (15, 16) and Shoulders (11, 12).
    2.  **Crossed Check:** If `LeftWrist.x < RightWrist.x` (assuming mirrored view), arms are crossed.
    3.  **Width Check:** If `abs(LeftWrist.x - RightWrist.x) < abs(LeftShoulder.x - RightShoulder.x)`, hands are tight inside the torso (Closed).
*   **Thresholds:**
    *   **Hands inside shoulders:** Closed / Defensive.
    *   **Hands outside shoulders:** Open / Confident.
*   **Implications:** Open posture increases perceived confidence and trustworthiness.
*   **Source:** *SapienAI* (Expressive Coherence) & *Biomechanics*.

---

### **4. Eye Contact Ratio**
*   **Definition:** The percentage of time the speaker is looking directly at the "audience" (the camera lens).
*   **Normalization:** **Eye Width**. Distance between Inner Eye Corner and Outer Eye Corner (Landmarks 33 vs 133 for MediaPipe FaceMesh).
*   **How to Calculate:**
    1.  Get **Iris Center** (Landmark 468).
    2.  Get **Eye Center** (Midpoint of corners).
    3.  `Deviation = distance(Iris, EyeCenter) / EyeWidth`.
*   **Thresholds:**
    *   **Deviation < 0.1:** Looking at Camera (Good).
    *   **Deviation > 0.2:** Looking Away (Reading/Distracted).
    *   **Target:** Maintain "Good" status for **> 60%** of the video.
*   **Implications:** Breaking eye contact signals nervousness or lack of preparation (reading notes).
*   **Source:** *SpeechMirror* (Gaze Estimation).

---

### **5. Volume Dynamics (Loudness)**
*   **Definition:** The variation in vocal loudness, distinguishing monotone speaking from dynamic projection.
*   **Normalization:** None. RMS (Root Mean Square) Amplitude is standard.
*   **How to Calculate:**
    1.  Load audio chunk into `librosa`.
    2.  `rms = librosa.feature.rms(y=audio)`
    3.  Calculate Standard Deviation: `std_dev = np.std(rms)`
*   **Thresholds:**
    *   **Std Dev < 0.02:** Monotone / Robotic.
    *   **Std Dev > 0.05:** **Dynamic / Engaging.**
*   **Implications:** Monotone voices cause listener fatigue. Variation keeps attention.
*   **Source:** *SpeechMirror* (Volatility) / *Librosa*.

---

### **6. Head Posture (Authority)**
*   **Definition:** The vertical alignment of the head, checking if the speaker is looking down (submissive/reading) or up (confident).
*   **Normalization:** **Neck Length**. Distance between Nose (0) and Shoulder Center (midpoint of 11 & 12).
*   **How to Calculate:**
    1.  `Vertical_Dist = (Shoulder_Y - Nose_Y)`
    2.  Compare to a calibrated "Neutral" neck length (captured at start) OR use heuristic ratio.
*   **Thresholds:**
    *   **Ratio < 0.10:** Head Down (Chin tucked).
    *   **Ratio > 0.15:** **Head Up (Neutral/Confident).**
*   **Implications:** Dropping the chin often correlates with reading notes or lack of confidence.
*   **Source:** *Biomechanics of Public Speaking* (Head Orientation).

---

### **7. Nervous Micro-Gestures**
*   **Definition:** Self-soothing behaviors like touching the neck, face, or playing with hair/rings.
*   **Normalization:** **Face Height**. Distance between Forehead (10) and Chin (152) in FaceMesh.
*   **How to Calculate:**
    1.  Calculate distance between Wrist (15 or 16) and Face Center (0).
    2.  If `Distance < (1.5 * FaceHeight)`, the hand is touching the face/neck.
*   **Thresholds:**
    *   **Duration > 2.0 seconds:** Flag as "Nervous Gesture."
    *   **Frequency > 3 times/min:** Flag as "High Anxiety."
*   **Implications:** These subconscious movements betray hidden stress even if the voice is calm.
*   **Source:** *SMG: Micro-gesture Dataset*.

---

### **8. Pause Duration**
*   **Definition:** The length of silence between spoken segments.
*   **Normalization:** None (Seconds).
*   **How to Calculate:**
    1.  Use Whisper segment timestamps.
    2.  `Gap = Segment_B_Start - Segment_A_End`.
*   **Thresholds:**
    *   **< 0.5s:** Rushing (Nervous).
    *   **0.5s – 1.5s:** **Natural / Emphasis.**
    *   **> 2.0s:** Awkward Silence / Lost train of thought.
*   **Implications:** Strategic pauses demonstrate mastery; awkward pauses demonstrate forgetting.
*   **Source:** *SpeechMirror* (Pace analysis).


### **9. Script Reading Detection (Sustained Head Down)**
*   **Definition:** Detecting when the user breaks eye contact by tilting their head downward for an extended period while speaking, indicating they are reading notes rather than presenting.
*   **Normalization:** **Ear-to-Nose Vertical Delta**. Compare the Y-coordinate of the Nose (0) against the Ears (7, 8).
*   **How to Calculate:**
    1.  Get **Nose Y** and average **Ear Y**.
    2.  `Head_Pitch = Nose_Y - Ear_Y`.
    3.  **AND Condition:** Check if `Audio Volume > Silence_Threshold` (Are they speaking?).
*   **Thresholds:**
    *   **Pitch Check:** If `Nose_Y` is significantly below `Ear_Y` (Positive value in MediaPipe coords).
    *   **Time Check:** State must persist for **> 3.0 seconds** (to distinguish from a quick nod).
*   **Implications:** "Reading" kills charisma. It tells the audience you don't know your material.
*   **Source:** *Biomechanics* (Posture) + *SpeechMirror* (Gaze Aversion logic).

---

### **10. Cognitive Load Indicator (The "Thinking" Gaze)**
*   **Definition:** Distinguishing between "looking at the audience" and the specific "looking up/away" behavior that humans do when trying to retrieve memory or form complex thoughts.
*   **Normalization:** **Eye Box**. The bounding box of the eye landmarks.
*   **How to Calculate:**
    1.  Calculate **Gaze Vector** (Iris position relative to eye center).
    2.  **Direction Classification:**
        *   **Up/Left or Up/Right:** "Accessing Memory" (Thinking).
        *   **Pure Left/Right:** "Distracted" (Looking at something else).
    3.  **AND Condition:** Often accompanied by **Filler Words** ("Umm...") or **Long Pauses**.
*   **Thresholds:**
    *   **Duration:** > 2.0 seconds (Brief glances are normal; staring at the ceiling is bad).
    *   **Frequency:** If this happens every sentence, the speaker is underprepared.
*   **Implications:** Occasional "thinking gaze" is natural. Constant looking away signals a lack of preparation or high cognitive stress.
*   **Source:** *SpeechMirror* (Eye Contact Analysis) & General Psychology (Eye Accessing Cues).

---

### Implementation Tip: The "State Machine"

To implement these, don't just check a single frame. Use a simple **State Machine** in your Python `MetricsCalculator`:

```python
class GazeInterpreter:
    def __init__(self):
        self.looking_down_frames = 0
        self.looking_up_frames = 0

    def analyze_frame(self, landmarks):
        # 1. Check Head Pitch
        if self.is_head_down(landmarks):
            self.looking_down_frames += 1
        else:
            self.looking_down_frames = 0
        
        # 2. Trigger Logic (30 frames = ~1 second)
        if self.looking_down_frames > 90: # 3 Seconds
            return "WARNING: You seem to be reading from a script."
            
        return "OK"
```