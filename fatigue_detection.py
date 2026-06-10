import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class BlinkDetector:
    def __init__(self, threshold=0.45, closed_frames_required=1, calibration_frames=45):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        
        # Eye landmarks
        # p1 (left corner), p2 (upper), p3 (upper), p4 (right corner), p5 (lower), p6 (lower)
        self.LEFT_EYE = [33, 159, 158, 133, 153, 145]
        self.RIGHT_EYE = [362, 386, 385, 263, 374, 380]
        
        self.threshold = threshold
        self.closed_frames_required = closed_frames_required
        self.calibration_frames = calibration_frames
        self.open_ear_samples = []
        self.baseline_ear = None
        self.smoothed_ear = None
        self.recent_ears = deque(maxlen=5)
        self.min_smoothed_ear = None
        self.blink_count = 0
        self.eyes_closed = False
        self.blink_detected = False
        self.blink_time = 0
        self.frame_count = 0
        self.closed_frames = 0
        
    def get_eye_aspect_ratio(self, eye_points):
        """Calculate EAR (Eye Aspect Ratio)"""
        p1 = np.array(eye_points[0])
        p2 = np.array(eye_points[1])
        p3 = np.array(eye_points[2])
        p4 = np.array(eye_points[3])
        p5 = np.array(eye_points[4])
        p6 = np.array(eye_points[5])
        
        vert1 = np.linalg.norm(p2 - p6)
        vert2 = np.linalg.norm(p3 - p5)
        horiz = np.linalg.norm(p1 - p4)
        
        ear = (vert1 + vert2) / (2.0 * horiz + 1e-6)
        return ear
    
    def process_frame(self, frame):
        """Process a single frame"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        left_ear = 0
        right_ear = 0
        text_info = "No face"
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye_points = [[landmarks[i].x, landmarks[i].y] for i in self.LEFT_EYE]
            right_eye_points = [[landmarks[i].x, landmarks[i].y] for i in self.RIGHT_EYE]
            
            left_ear = self.get_eye_aspect_ratio(left_eye_points)
            right_ear = self.get_eye_aspect_ratio(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2

            # Smooth EAR to reduce jitter
            self.recent_ears.append(avg_ear)
            self.smoothed_ear = float(np.mean(self.recent_ears))
            if self.min_smoothed_ear is None or self.smoothed_ear < self.min_smoothed_ear:
                self.min_smoothed_ear = self.smoothed_ear

            # Calibrate baseline using open-eye samples
            if self.baseline_ear is None:
                self.open_ear_samples.append(self.smoothed_ear)
                if len(self.open_ear_samples) >= self.calibration_frames:
                    self.baseline_ear = float(np.median(self.open_ear_samples))
                    # Adaptive threshold based on baseline
                    self.threshold = max(0.25, self.baseline_ear * 0.92)

            is_closed = self.smoothed_ear < self.threshold if self.baseline_ear is not None else False
            
            # Debug output every 15 frames
            if self.frame_count % 15 == 0:
                status = "CLOSED" if is_closed else "OPEN"
                base = f"{self.baseline_ear:.3f}" if self.baseline_ear is not None else "-"
                min_ear = f"{self.min_smoothed_ear:.3f}" if self.min_smoothed_ear is not None else "-"
                print(
                    f"Frame {self.frame_count}: L={left_ear:.3f} R={right_ear:.3f} "
                    f"Avg={avg_ear:.3f} Smoothed={self.smoothed_ear:.3f} "
                    f"[{status}] Threshold={self.threshold:.3f} Baseline={base} Min={min_ear}",
                    flush=True
                )
            
            # Blink detection with required consecutive closed frames
            if is_closed:
                self.closed_frames += 1
            else:
                if self.closed_frames >= self.closed_frames_required:
                    self.blink_count += 1
                    self.blink_detected = True
                    self.blink_time = time.time()
                    print(f"\n>>> BLINK #{self.blink_count} DETECTED! <<<\n", flush=True)
                self.closed_frames = 0

            self.eyes_closed = is_closed
            text_info = f"L={left_ear:.2f} R={right_ear:.2f} A={self.smoothed_ear:.2f}"
        
        # Clear blink flag
        if self.blink_detected and time.time() - self.blink_time > 0.5:
            self.blink_detected = False
        
        return frame, left_ear, right_ear, text_info


def main():
    print("=" * 60)
    print("BLINK DETECTION SYSTEM - Starting...")
    print("=" * 60)
    print("Press 'Q' to quit | Press 'R' to reset counter\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    detector = BlinkDetector(threshold=0.75)
    print("Camera is ready. Face must be clearly visible.\n")
    
    fps_start = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        frame, left_ear, right_ear, info = detector.process_frame(frame)
        
        # Clean UI: show only eye status
        status_color = (0, 0, 255) if detector.eyes_closed else (0, 255, 0)
        status_text = "EYES: CLOSED" if detector.eyes_closed else "EYES: OPEN"
        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        cv2.imshow("Blink Detection System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\nExiting program...")
            break
        elif key == ord('r') or key == ord('R'):
            detector.blink_count = 0
            print("Blink counter has been reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed. Goodbye!")


if __name__ == "__main__":
    main()
