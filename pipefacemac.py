# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
import sys
import platform
import traceback

# Ensure filterpy import is present and correct
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    print("---------------------------------------------------------")
    print("Error: filterpy library not found or Q_discrete_white_noise not available.")
    print("Please install or update filterpy:")
    print("  pip install filterpy")
    print("---------------------------------------------------------")
    sys.exit(1)


class HeadGazeTracker:
    def __init__(self):
        print("Initializing HeadGazeTracker...")
        if platform.system() == "Darwin":
             print("macOS detected. Ensure necessary permissions (Accessibility, Camera) are granted.")
        elif platform.system() == "Windows":
             print("Windows detected.")
        else:
             print(f"{platform.system()} detected.")

        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6, # Keep reasonable confidence
            min_tracking_confidence=0.6
        )

        # Eye landmarks for EAR
        self.LEFT_EYE_V = [159, 145]; self.RIGHT_EYE_V = [386, 374]
        self.LEFT_EYE_H = [33, 133]; self.RIGHT_EYE_H = [362, 263]

        # Screen dimensions with fallback
        try:
            self.screen_width, self.screen_height = pyautogui.size()
            print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        except Exception as e:
             print(f"Warning: Failed to get screen size: {e}. Defaulting to 1920x1080.")
             self.screen_width, self.screen_height = 3024, 1964

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

        # Calibration settings
        self.calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        self.calibration_data = []
        self.calibration_complete = False
        self.calibration_matrix = None
        self.reference_head_pose = None # (raw_neutral_yaw, raw_neutral_pitch)

        # --- Kalman Filter Setup ---
        self.kf = KalmanFilter(dim_x=4, dim_z=2) # State: [x, y, vx, vy]
        self.kf.x = np.array([[self.screen_width / 2], [self.screen_height / 2], [0.], [0.]]) # (4, 1) vector
        self.kf.F = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.kf.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

        # *** TUNING: Increased measurement noise, slightly decreased process noise ***
        measurement_noise_std = 30.0 # TUNABLE: Increase if cursor is jittery/jumps with measurement
        self.kf.R = np.diag([measurement_noise_std**2, measurement_noise_std**2])
        self.process_noise_std = 3.0 # TUNABLE: Decrease if cursor overshoots/oscillates

        self.kf.Q = np.eye(4) # Placeholder, calculated dynamically using process_noise_std
        self.kf.P = np.eye(4) * 500.
        self.last_time = time.time()

        # --- Wink Detection ---
        self.EAR_THRESHOLD = 0.20; self.WINK_CONSEC_FRAMES = 2
        self.CLICK_COOLDOWN = 0.5; self.DOUBLE_CLICK_WINDOW = 0.4
        self.left_wink_counter = 0; self.right_wink_counter = 0
        self.left_eye_closed = False; self.right_eye_closed = False
        self.last_left_click_time = 0; self.last_right_click_time = 0
        self.last_left_wink_detect_time = 0

        # --- Raw Pose Smoothing (Optional) ---
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.pose_smoothing_factor = 0.6 # TUNABLE: 0 = no smoothing, closer to 1 = more smoothing
        self.first_pose = True

        print("Initialization complete.")

    # _calculate_distance and _calculate_ear remain the same as the previous version
    def _calculate_distance(self, p1, p2):
        if p1 is None or p2 is None: return 0.0
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _calculate_ear(self, landmarks, eye_v_indices, eye_h_indices):
        try:
            if not landmarks or max(eye_v_indices + eye_h_indices) >= len(landmarks):
                 return 0.3
            p_top = landmarks[eye_v_indices[0]]; p_bottom = landmarks[eye_v_indices[1]]
            p_left = landmarks[eye_h_indices[0]]; p_right = landmarks[eye_h_indices[1]]
            v_dist = self._calculate_distance(p_top, p_bottom)
            h_dist = self._calculate_distance(p_left, p_right)
            if h_dist < 1e-6: return 0.3
            ear = v_dist / h_dist
            return ear
        except Exception: return 0.3 # Simplified error handling

    # _detect_winks remains the same as the previous version
    def _detect_winks(self, landmarks):
        current_time = time.time()
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_V, self.LEFT_EYE_H)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_V, self.RIGHT_EYE_H)
        ear_thresh_low = self.EAR_THRESHOLD - 0.02
        ear_thresh_high = self.EAR_THRESHOLD + 0.02
        self.left_eye_closed = left_ear < ear_thresh_low
        self.right_eye_closed = right_ear < ear_thresh_low

        # Left Wink / Double Click
        if self.left_eye_closed and right_ear > ear_thresh_high: self.left_wink_counter += 1
        else: self.left_wink_counter = 0
        if self.left_wink_counter == self.WINK_CONSEC_FRAMES and \
           current_time - self.last_left_click_time > self.CLICK_COOLDOWN:
            time_since_last_detect = current_time - self.last_left_wink_detect_time
            if time_since_last_detect < self.DOUBLE_CLICK_WINDOW:
                print(f"Double Left Click Triggered! (dT={time_since_last_detect:.2f}s)")
                try: pyautogui.doubleClick(button='left')
                except Exception as e: print(f"PyAutoGUI Error: {e}")
                self.last_left_click_time = current_time; self.last_left_wink_detect_time = 0
            else:
                print(f"Left Click Triggered! (dT={time_since_last_detect:.2f}s)")
                try: pyautogui.click(button='left')
                except Exception as e: print(f"PyAutoGUI Error: {e}")
                self.last_left_click_time = current_time; self.last_left_wink_detect_time = current_time
            self.left_wink_counter = 0
        elif self.left_wink_counter > self.WINK_CONSEC_FRAMES * 2 : self.left_wink_counter = 0

        # Right Wink
        if self.right_eye_closed and left_ear > ear_thresh_high: self.right_wink_counter += 1
        else: self.right_wink_counter = 0
        if self.right_wink_counter == self.WINK_CONSEC_FRAMES and \
           current_time - self.last_right_click_time > self.CLICK_COOLDOWN:
             print("Right Click Triggered!")
             try: pyautogui.click(button='right')
             except Exception as e: print(f"PyAutoGUI Error: {e}")
             self.last_right_click_time = current_time; self.right_wink_counter = 0
        elif self.right_wink_counter > self.WINK_CONSEC_FRAMES * 2: self.right_wink_counter = 0

        return left_ear, right_ear

    def _get_head_pose(self, landmarks, image_shape):
        """Estimate head yaw and pitch, with stabilization."""
        h, w = image_shape[:2]
        required_indices = [1, 152, 133, 362, 10, 33, 263]
        if not landmarks or not all(idx < len(landmarks) for idx in required_indices):
            ref_pose = self.reference_head_pose if self.reference_head_pose else (0.0, 0.0)
            return ref_pose[0], ref_pose[1] # Return neutral/default if landmarks missing

        try:
            nose_tip = landmarks[1]; chin = landmarks[152]
            left_eye_inner = landmarks[133]; right_eye_inner = landmarks[362]
            forehead = landmarks[10]

            # --- Yaw ---
            nose_x = nose_tip.x * w
            left_eye_x = left_eye_inner.x * w; right_eye_x = right_eye_inner.x * w
            eye_center_x = (left_eye_x + right_eye_x) / 2.0
            interocular_dist = abs(right_eye_x - left_eye_x)
            # *** STABILIZATION: Prevent division by very small number ***
            if interocular_dist < 10: interocular_dist = 10 # Assume min pixel distance

            nose_offset_x = nose_x - eye_center_x
            # *** TUNING: Slightly reduced sensitivity ***
            head_yaw = (nose_offset_x / interocular_dist) * 1.3 # TUNABLE
            head_yaw = np.clip(head_yaw, -1.0, 1.0)

            # --- Pitch ---
            nose_y = nose_tip.y * h; chin_y = chin.y * h; forehead_y = forehead.y * h
            eye_center_y = (left_eye_inner.y + right_eye_inner.y) / 2.0 * h
            face_height = abs(chin_y - forehead_y)
            # *** STABILIZATION: Prevent division by very small number ***
            if face_height < 20: face_height = 20 # Assume min pixel height

            nose_offset_y = (nose_y - eye_center_y) / face_height
            neutral_pitch_offset = 0.1 # TUNABLE
            # *** TUNING: Slightly reduced sensitivity ***
            head_pitch = (nose_offset_y - neutral_pitch_offset) * 2.2 # TUNABLE
            head_pitch = np.clip(head_pitch, -0.8, 0.8)

            # --- Optional Smoothing ---
            if self.first_pose:
                self.smooth_yaw = head_yaw
                self.smooth_pitch = head_pitch
                self.first_pose = False
            else:
                self.smooth_yaw = self.pose_smoothing_factor * self.smooth_yaw + (1 - self.pose_smoothing_factor) * head_yaw
                self.smooth_pitch = self.pose_smoothing_factor * self.smooth_pitch + (1 - self.pose_smoothing_factor) * head_pitch

            # Return the smoothed values
            return self.smooth_yaw, self.smooth_pitch

        except Exception as e:
             # print(f"Error getting head pose: {e}") # Reduce console spam
             ref_pose = self.reference_head_pose if self.reference_head_pose else (0.0, 0.0)
             # Return last known smoothed pose or reference pose on error
             return self.smooth_yaw if not self.first_pose else ref_pose[0], \
                    self.smooth_pitch if not self.first_pose else ref_pose[1]


    # run_calibration and _compute_calibration_matrix remain the same as previous version
    def run_calibration(self, cap):
        print("\n--- Starting Calibration ---")
        print("Instructions: Follow the green dot with your head.")
        print("First, look straight at the center dot for neutral pose.")
        self.calibration_data = []; self.reference_head_pose = None; self.calibration_complete = False
        calib_win_name = "Calibration Target"; feed_win_name = "Camera Feed (Calibration)"
        cv2.namedWindow(calib_win_name, cv2.WINDOW_NORMAL); cv2.namedWindow(feed_win_name, cv2.WINDOW_NORMAL)
        try:
            calib_win_w = max(300, self.screen_width // 3); calib_win_h = max(300, self.screen_height // 3)
            cv2.resizeWindow(calib_win_name, calib_win_w, calib_win_h)
            cv2.moveWindow(calib_win_name, self.screen_width // 2 - calib_win_w // 2, self.screen_height // 2 - calib_win_h // 2)
            feed_w, feed_h = 320, 240; cv2.resizeWindow(feed_win_name, feed_w, feed_h); cv2.moveWindow(feed_win_name, 30, 30)
        except Exception as e: print(f"Warning: Could not resize/move calibration windows: {e}"); calib_win_w, calib_win_h = 400, 400

        # --- Neutral Pose ---
        print("\nStep 1: Neutral Pose Calibration"); print("Look DIRECTLY at the green dot. Keep still...")
        neutral_data_raw = []; calib_img = np.zeros((calib_win_h, calib_win_w, 3), dtype=np.uint8)
        center_x, center_y = calib_win_w // 2, calib_win_h // 2
        cv2.circle(calib_img, (center_x, center_y), 20, (0, 0, 255), 2); cv2.circle(calib_img, (center_x, center_y), 8, (0, 255, 0), -1)
        cv2.putText(calib_img, "Look Here (Neutral)", (center_x - 100, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(calib_win_name, calib_img); cv2.waitKey(1); time.sleep(1.5)
        start_time = time.time(); collection_duration = 3.0; min_neutral_samples = 15; self.first_pose = True # Reset smoother
        while time.time() - start_time < collection_duration:
            ret, frame = cap.read();
            if not ret: continue
            frame = cv2.flip(frame, 1); frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False; results = self.face_mesh.process(frame_rgb); frame.flags.writeable = True
            progress_img = calib_img.copy(); remaining_time = collection_duration - (time.time() - start_time)
            cv2.putText(progress_img, f"Keep Still: {remaining_time:.1f}s", (center_x - 50, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(calib_win_name, progress_img)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                try:
                    # Get raw pose, don't use smoother during calibration capture
                    head_yaw, head_pitch = self._get_head_pose(landmarks, frame.shape) # Call it to update smoother state if needed, but use raw below
                    raw_yaw = head_yaw # Actually use the direct output for calibration avg
                    raw_pitch = head_pitch
                    if not self.first_pose: # Discard first frame reading if smoother was just reset
                         neutral_data_raw.append((raw_yaw, raw_pitch))
                    self._draw_landmarks(frame, landmarks)
                except Exception as e: cv2.putText(frame, "Pose Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, "Neutral Pose Capture", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1); cv2.imshow(feed_win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): print("Calibration cancelled."); cv2.destroyAllWindows(); return False
        if len(neutral_data_raw) >= min_neutral_samples:
            avg_yaw = np.mean([d[0] for d in neutral_data_raw]); avg_pitch = np.mean([d[1] for d in neutral_data_raw])
            self.reference_head_pose = (avg_yaw, avg_pitch); print(f"Neutral calibrated: Raw Yaw={avg_yaw:.3f}, Raw Pitch={avg_pitch:.3f}")
        else: print(f"Failed neutral samples ({len(neutral_data_raw)}/{min_neutral_samples}). Try again."); cv2.destroyAllWindows(); return False

        # --- Calibration Points ---
        print("\nStep 2: Calibration Points"); print("Look at the green dot as it moves...")
        collection_duration_point = 1.5; min_point_samples = 5
        for i, (x_ratio, y_ratio) in enumerate(self.calibration_points):
            target_x = int(x_ratio * calib_win_w); target_y = int(y_ratio * calib_win_h)
            print(f"Point {i+1}/{len(self.calibration_points)}: Look at target ({x_ratio:.1f}, {y_ratio:.1f})...")
            point_data_raw = []; start_time = time.time(); self.first_pose = True # Reset smoother for each point
            while time.time() - start_time < collection_duration_point:
                ret, frame = cap.read();
                if not ret: continue
                frame = cv2.flip(frame, 1); frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False; results = self.face_mesh.process(frame_rgb); frame.flags.writeable = True
                point_img = np.zeros((calib_win_h, calib_win_w, 3), dtype=np.uint8)
                cv2.circle(point_img, (target_x, target_y), 20, (0, 0, 255), 2); cv2.circle(point_img, (target_x, target_y), 8, (0, 255, 0), -1)
                cv2.putText(point_img, f"Look Here ({i+1})", (target_x - 50 if target_x > 50 else 5, target_y - 30 if target_y > 30 else 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                remaining_time = collection_duration_point - (time.time() - start_time)
                cv2.putText(point_img, f"{remaining_time:.1f}s", (target_x - 20 if target_x > 20 else 5, target_y + 40 if target_y < calib_win_h - 50 else calib_win_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(calib_win_name, point_img)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    try:
                        # Get raw pose, don't use smoother during calibration capture
                        head_yaw, head_pitch = self._get_head_pose(landmarks, frame.shape) # Call to update smoother state if needed
                        raw_yaw = head_yaw; raw_pitch = head_pitch
                        if not self.first_pose: # Discard first frame
                             point_data_raw.append((raw_yaw, raw_pitch))
                        self._draw_landmarks(frame, landmarks)
                    except Exception as e: cv2.putText(frame, "Pose Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(frame, f"Point {i+1} Capture", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1); cv2.imshow(feed_win_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): print("Calibration cancelled."); cv2.destroyAllWindows(); return False
            if len(point_data_raw) >= min_point_samples:
                avg_yaw = np.mean([p[0] for p in point_data_raw]); avg_pitch = np.mean([p[1] for p in point_data_raw])
                self.calibration_data.append((avg_yaw, avg_pitch, x_ratio, y_ratio))
                print(f"  Point {i+1} recorded (RawYaw={avg_yaw:.3f}, RawPitch={avg_pitch:.3f}) -> Target ({x_ratio:.1f}, {y_ratio:.1f})")
            else: print(f"  Skipped point {i+1}: Insufficient samples ({len(point_data_raw)}/{min_point_samples}).")

        # --- Finalize ---
        cv2.destroyWindow(calib_win_name); cv2.destroyWindow(feed_win_name)
        print("\n--- Computing Calibration Mapping ---")
        if len(self.calibration_data) >= 6:
            if self._compute_calibration_matrix():
                print("Calibration successful!"); self.calibration_complete = True
                self.kf.x = np.array([[self.screen_width / 2], [self.screen_height / 2], [0.], [0.]]); self.kf.P = np.eye(4) * 500.
                self.last_time = time.time(); self.left_wink_counter, self.right_wink_counter = 0, 0
                self.last_left_click_time, self.last_right_click_time = 0, 0; self.last_left_wink_detect_time = 0
                self.first_pose = True # Reset smoother for tracking
                return True
            else: print("Calibration matrix computation failed."); self.calibration_complete = False; return False
        else: print(f"Calibration failed: Not enough points ({len(self.calibration_data)}). Need 6+."); self.calibration_complete = False; return False

    def _compute_calibration_matrix(self):
        if not self.calibration_data or self.reference_head_pose is None: return False
        ref_yaw, ref_pitch = self.reference_head_pose
        rel_yaw = np.array([p[0] - ref_yaw for p in self.calibration_data])
        rel_pitch = np.array([p[1] - ref_pitch for p in self.calibration_data])
        target_x_ratio = np.array([p[2] for p in self.calibration_data])
        target_y_ratio = np.array([p[3] for p in self.calibration_data])
        feature_matrix = np.column_stack([rel_yaw, rel_pitch, rel_yaw**2, rel_pitch**2, rel_yaw * rel_pitch, np.ones(len(rel_yaw))])
        try:
            x_coeffs, _, _, _ = np.linalg.lstsq(feature_matrix, target_x_ratio, rcond=None)
            y_coeffs, _, _, _ = np.linalg.lstsq(feature_matrix, target_y_ratio, rcond=None)
            self.calibration_matrix = (x_coeffs, y_coeffs); print("Calibration matrix computed."); return True
        except Exception as e: print(f"Error computing calibration matrix: {e}"); self.calibration_matrix = None; return False

    def map_head_to_screen(self, head_yaw, head_pitch):
        """Maps relative head pose to screen coordinates, with clamping."""
        if not self.calibration_complete or self.calibration_matrix is None or self.reference_head_pose is None:
            # Fallback: return current KF state if not calibrated
            return self.kf.x[0, 0], self.kf.x[1, 0]

        ref_yaw, ref_pitch = self.reference_head_pose
        x_coeffs, y_coeffs = self.calibration_matrix

        # Use the smoothed pose from _get_head_pose for mapping
        rel_yaw = head_yaw - ref_yaw
        rel_pitch = head_pitch - ref_pitch

        features = np.array([rel_yaw, rel_pitch, rel_yaw**2, rel_pitch**2, rel_yaw * rel_pitch, 1.0])
        screen_x_ratio = np.dot(features, x_coeffs)
        screen_y_ratio = np.dot(features, y_coeffs)

        # Optional: Print raw ratios for debugging
        # print(f"Raw Ratios: X={screen_x_ratio:.3f}, Y={screen_y_ratio:.3f}")

        # Clamp ratios slightly outside 0-1 before converting to pixels
        screen_x_ratio = np.clip(screen_x_ratio, -0.1, 1.1)
        screen_y_ratio = np.clip(screen_y_ratio, -0.1, 1.1)

        screen_x = screen_x_ratio * self.screen_width
        screen_y = screen_y_ratio * self.screen_height

        # *** IMPORTANT: Clamp final pixel coords BEFORE returning for KF ***
        # Add a small margin to prevent hitting the absolute edge
        margin = 1
        screen_x = max(margin, min(self.screen_width - 1 - margin, screen_x))
        screen_y = max(margin, min(self.screen_height - 1 - margin, screen_y))

        return screen_x, screen_y

    # _draw_landmarks remains the same as previous version
    def _draw_landmarks(self, frame, landmarks):
        if not landmarks: return
        h, w = frame.shape[:2]
        left_color = (0, 255, 255) if self.left_eye_closed else (255, 255, 0)
        right_color = (0, 255, 255) if self.right_eye_closed else (255, 255, 0)
        for idx in self.LEFT_EYE_V + self.LEFT_EYE_H:
             if idx < len(landmarks): cv2.circle(frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, left_color, -1)
        for idx in self.RIGHT_EYE_V + self.RIGHT_EYE_H:
             if idx < len(landmarks): cv2.circle(frame, (int(landmarks[idx].x * w), int(landmarks[idx].y * h)), 2, right_color, -1)
        if 1 < len(landmarks): cv2.circle(frame, (int(landmarks[1].x * w), int(landmarks[1].y * h)), 3, (0, 0, 255), -1)


    def start_tracking(self):
        """Main tracking loop."""
        cap = None; camera_found = False
        preferred_indices = [0, 1]; other_indices = [i for i in range(2, 6)]
        for i in preferred_indices + other_indices:
            print(f"Attempting camera index: {i}")
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                 print(f"  Camera {i} opened."); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                 time.sleep(0.5); ret, _ = cap.read(); actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH); actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                 if ret and actual_width > 0 and actual_height > 0: print(f"  Camera {i} ready. Res: {int(actual_width)}x{int(actual_height)}"); camera_found = True; break
                 else: print(f"  Camera {i} failed read/res check. Closing."); cap.release(); cap = None
            else: print(f"  Failed to open camera {i}.");
            if cap: cap.release(); cap = None
        if not camera_found or cap is None: print("Error: Could not open webcam."); return

        if not self.run_calibration(cap): print("Calibration failed/aborted. Exiting."); cap.release(); cv2.destroyAllWindows(); return

        print("\n--- Starting Tracking ---")
        main_win_name = 'Head Tracking Feed'; cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL)
        feed_w, feed_h = 480, 360; cv2.resizeWindow(main_win_name, feed_w, feed_h)
        try: cv2.moveWindow(main_win_name, 30, self.screen_height - feed_h - 50)
        except: cv2.moveWindow(main_win_name, 30, 30)

        frame_count = 0; fps = 0; fps_start_time = time.time(); self.last_time = time.time()
        self.first_pose = True # Ensure smoother resets for tracking start

        while True:
            current_time = time.time(); dt = current_time - self.last_time
            if dt <= 1e-6 : dt = 1/30.0
            elif dt > 0.5 : dt = 1/30.0 # Reset dt after pause/lag
            self.last_time = current_time

            ret, frame = cap.read()
            if not ret: print("Warning: Failed frame read."); time.sleep(0.1); continue
            frame = cv2.flip(frame, 1); frame_height, frame_width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_rgb.flags.writeable = False
            try: results = self.face_mesh.process(frame_rgb)
            except Exception as e: print(f"MediaPipe Error: {e}"); frame.flags.writeable = True; continue
            frame.flags.writeable = True

            left_ear, right_ear = 0.3, 0.3

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                try:
                    # 1. Get Smoothed Head Pose
                    head_yaw, head_pitch = self._get_head_pose(landmarks, frame.shape)

                    # 2. Map to Screen Coords (Measurement, clamped)
                    measured_x, measured_y = self.map_head_to_screen(head_yaw, head_pitch)
                    measurement = np.array([[measured_x], [measured_y]])

                    # Optional: Print measurement fed to KF
                    # print(f"Measurement -> KF: ({measured_x:.1f}, {measured_y:.1f})")

                    # 3. Kalman Filter Update
                    self.kf.F[0, 2] = dt; self.kf.F[1, 3] = dt
                    q_block_half = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise_std**2)
                    self.kf.Q = np.zeros((4, 4))
                    self.kf.Q[0,0] = self.kf.Q[1,1] = q_block_half[0,0]; self.kf.Q[2,2] = self.kf.Q[3,3] = q_block_half[1,1]
                    self.kf.Q[0,2] = self.kf.Q[2,0] = q_block_half[0,1]; self.kf.Q[1,3] = self.kf.Q[3,1] = q_block_half[0,1]

                    self.kf.predict()
                    self.kf.update(measurement)

                    # 4. Get Filtered State & Clamp (final safety)
                    filtered_x = self.kf.x[0, 0]; filtered_y = self.kf.x[1, 0]
                    margin = 0 # Can use 0 here as map_head_to_screen already added margin
                    smooth_x = max(margin, min(self.screen_width - 1 - margin, filtered_x))
                    smooth_y = max(margin, min(self.screen_height - 1 - margin, filtered_y))

                    # 5. Move Mouse
                    try: pyautogui.moveTo(smooth_x, smooth_y, duration=0.0)
                    except Exception as e: print(f"PyAutoGUI Error: {e}")

                    # 6. Detect Winks
                    left_ear, right_ear = self._detect_winks(landmarks)

                    # 7. Draw Visuals
                    self._draw_landmarks(frame, landmarks)
                    cv2.putText(frame, f"Pos: ({int(smooth_x)}, {int(smooth_y)})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    print(f"Error in main loop processing: {e}"); traceback.print_exc()
                    cv2.putText(frame, "Processing Error", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            else: # No face detected
                # *** IMPROVED HANDLING: Freeze cursor and reset KF velocity ***
                cv2.putText(frame, "Face not detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                # Option 1: Freeze cursor (safer) - Do nothing with pyautogui.moveTo
                # Option 2: Reset KF velocity to prevent runaway prediction if coasting were enabled
                self.kf.x[2, 0] = 0.0 # Reset vx
                self.kf.x[3, 0] = 0.0 # Reset vy
                # Reset smoother state as well
                self.first_pose = True
                # Reset wink states
                self.left_wink_counter, self.right_wink_counter = 0, 0
                self.left_eye_closed, self.right_eye_closed = False, False
                self.last_left_wink_detect_time = 0

            # --- Display Info ---
            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                 fps = frame_count / (time.time() - fps_start_time); frame_count = 0; fps_start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"L EAR: {left_ear:.2f}", (10, frame_height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"R EAR: {right_ear:.2f}", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "Q: Quit | C: Recalibrate", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(main_win_name, frame)

            # --- Handle Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("Quitting..."); break
            elif key == ord('c'):
                print("Recalibrating..."); cv2.destroyWindow(main_win_name)
                if not self.run_calibration(cap): print("Recalibration failed/aborted. Exiting."); break
                else: # Recreate tracking window
                    print("\n--- Resuming Tracking ---"); cv2.namedWindow(main_win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(main_win_name, feed_w, feed_h);
                    try: cv2.moveWindow(main_win_name, 30, self.screen_height - feed_h - 50)
                    except: cv2.moveWindow(main_win_name, 30, 30)
                    frame_count = 0; fps_start_time = time.time(); self.last_time = time.time()
                    self.first_pose = True # Reset smoother

        # --- Cleanup ---
        print("Releasing camera and closing windows.");
        if cap: cap.release(); cv2.destroyAllWindows(); print("Program finished.")


if __name__ == "__main__":
    print("-------------------------------------")
    print(" Head Gaze Tracker Initializing... ")
    print("-------------------------------------")
    try:
        import cv2; import mediapipe; import numpy; import pyautogui; import filterpy
        print("Core dependencies found.")
    except ImportError as e: print(f"Error: Missing dependency - {e.name}"); sys.exit(1)

    tracker = HeadGazeTracker()
    tracker.start_tracking()