import math
import os
import random
import time

import cv2
import numpy as np
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseController:
    def __init__(self, smooth_alpha=0.25):
        try:
            base_options = python.BaseOptions(model_asset_path=self._get_model_path())
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as _rm
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=_rm.VisionTaskRunningMode.VIDEO,
                output_segmentation_masks=False,
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
        except FileNotFoundError:
            print("Warning: Could not load pose model, downloading...")
            # Download the model if not present
            self._download_model()
            base_options = python.BaseOptions(model_asset_path=self._get_model_path())
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as _rm
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=_rm.VisionTaskRunningMode.VIDEO,
                output_segmentation_masks=False,
            )
            self.pose = vision.PoseLandmarker.create_from_options(options)
        
        self.smooth_alpha = smooth_alpha
        self.filtered_level = 0.0

    def _download_model(self):
        """Download the PoseLandmarker model if needed."""
        import urllib.request
        model_path = self._get_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        try:
            print(f"Downloading model to {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully")
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Fallback: use a lighter fallback approach
            raise

    def _get_model_path(self):
        """Get the path to the PoseLandmarker model."""
        import mediapipe
        models_dir = os.path.join(
            os.path.dirname(mediapipe.__file__),
            "tasks", "python", "vision"
        )
        model_path = os.path.join(models_dir, "pose_landmarker_lite.task")
        return model_path

    def process(self, frame_bgr):
        # Convert to RGB numpy array for the pose landmarker.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # The Task API may expose different Image helpers across versions; use
        # a robust fallback: if the landmarker supports `detect_for_video`
        # (which accepts numpy arrays/frames), call it; otherwise try the
        # `detect` path but construct a simple shim object.
        results = None
        try:
            # Construct a mediapipe Image object from the core.image API
            from mediapipe.tasks.python.vision.core import image as mp_image
            mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, frame_rgb)

            # Prefer positional timestamp for detect_for_video (APIs vary).
            if hasattr(self.pose, "detect_for_video"):
                try:
                    results = self.pose.detect_for_video(mp_img, int(time.time() * 1000))
                except TypeError:
                    # Some builds expect only the image argument.
                    results = self.pose.detect_for_video(mp_img)
            else:
                results = self.pose.detect(mp_img)
        except Exception:
            # As a last resort, try passing raw numpy array to detect
            try:
                results = self.pose.detect(frame_rgb)
            except Exception:
                raise
        arm_level = 0.0
        detected = False

        # Normalize access to pose landmarks depending on the API shape.
        pose_landmarks = None
        if results is None:
            pose_landmarks = None
        elif hasattr(results, "pose_landmarks") and results.pose_landmarks:
            pose_landmarks = results.pose_landmarks
        elif isinstance(results, (list, tuple)) and results:
            # sometimes detect returns a list of detections
            pose_landmarks = results[0].pose_landmarks if hasattr(results[0], 'pose_landmarks') else None

        if pose_landmarks:
            detected = True
            # pose_landmarks may be a wrapper; try to index it consistently
            lm = pose_landmarks[0] if isinstance(pose_landmarks, (list, tuple)) else pose_landmarks
            arm_level = self._arm_level_from_landmarks(lm)
            try:
                h, w = frame_bgr.shape[:2]
                self._draw_landmarks(frame_bgr, lm, w, h)
            except Exception:
                pass

        self.filtered_level = (
            self.smooth_alpha * arm_level + (1.0 - self.smooth_alpha) * self.filtered_level
        )
        return frame_bgr, self.filtered_level, detected

    def _draw_landmarks(self, image, landmarks, frame_width, frame_height):
        """Draw pose landmarks on the image."""
        # Draw connections (skeleton)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
        ]
        h, w = image.shape[:2]
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                if start.visibility > 0.5 and end.visibility > 0.5:
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks as circles
        for i, landmark in enumerate(landmarks):
            if landmark.visibility > 0.5:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    def _arm_level_from_landmarks(self, landmarks):
        sides = [
            (11, 13, 15, 23),  # Left shoulder, elbow, wrist, hip
            (12, 14, 16, 24),  # Right shoulder, elbow, wrist, hip
        ]
        levels = []
        for shoulder_id, elbow_id, wrist_id, hip_id in sides:
            shoulder = landmarks[shoulder_id]
            elbow = landmarks[elbow_id]
            wrist = landmarks[wrist_id]
            hip = landmarks[hip_id]
            if min(shoulder.visibility, elbow.visibility, wrist.visibility, hip.visibility) < 0.5:
                continue
            level = max(
                self._arm_angle_level(shoulder, elbow, hip),
                self._arm_angle_level(shoulder, wrist, hip),
            )
            levels.append(level)
        if not levels:
            return 0.0
        return min(1.0, sum(levels) / len(levels))

    def _arm_angle_level(self, shoulder, joint, hip):
        arm_vec = (joint.x - shoulder.x, joint.y - shoulder.y)
        torso_vec = (hip.x - shoulder.x, hip.y - shoulder.y)
        arm_len = math.hypot(*arm_vec)
        torso_len = math.hypot(*torso_vec)
        if arm_len < 1e-4 or torso_len < 1e-4:
            return 0.0
        dot = arm_vec[0] * torso_vec[0] + arm_vec[1] * torso_vec[1]
        cos_angle = max(-1.0, min(1.0, dot / (arm_len * torso_len)))
        angle_deg = math.degrees(math.acos(cos_angle))
        level = (angle_deg - 20.0) / 70.0
        return max(0.0, min(1.0, level))


class RepDetector:
    def __init__(self, trigger_threshold=0.9, reset_threshold=0.7):
        self.trigger_threshold = trigger_threshold
        self.reset_threshold = reset_threshold
        self.armed = True

    def update(self, arm_level, detected):
        if not detected:
            self.armed = True
            return False
        if self.armed and arm_level >= self.trigger_threshold:
            self.armed = False
            return True
        if not self.armed and arm_level <= self.reset_threshold:
            self.armed = True
        return False


class FlappyGame:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pose Flappy Bird")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.bird_frames = self._load_bird_frames()
        self.bird_frame_index = 0
        self.bird_anim_time = 0.0
        self.bird_anim_fps = 10.0
        self.flap_cooldown = 0.22
        self.reset()

    def reset(self):
        self.bird_x = 120
        self.bird_y = self.height / 2
        self.bird_vel = 0.0
        self.bird_frame_index = 0
        self.bird_anim_time = 0.0
        self.last_flap_time = 0.0
        self.gravity = 470.0
        self.flap_strength = 300.0
        self.pipes = []
        self.pipe_gap = 160
        self.pipe_width = 70
        self.pipe_spacing = 320
        self.pipe_speed = 120.0
        self.last_pipe_x = self.width
        self.game_over = False
        self.game_over_time = 0.0

    def tick(self, arm_level, flap):
        dt = self.clock.tick(60) / 1000.0
        if self.game_over:
            if time.time() - self.game_over_time > 2.0:
                self.reset()
            return dt

        now = time.time()
        if flap and (now - self.last_flap_time) >= self.flap_cooldown:
            self.bird_vel = -self.flap_strength
            self.last_flap_time = now
        self.bird_vel += self.gravity * dt
        self.bird_y += self.bird_vel * dt
        self._update_animation(dt)

        self._update_pipes(dt)
        self._check_collisions()
        return dt

    def _update_animation(self, dt):
        if not self.bird_frames:
            return
        self.bird_anim_time += dt
        frame_count = len(self.bird_frames)
        if self.bird_anim_time >= 1.0 / self.bird_anim_fps:
            steps = int(self.bird_anim_time * self.bird_anim_fps)
            self.bird_frame_index = (self.bird_frame_index + steps) % frame_count
            self.bird_anim_time -= steps / self.bird_anim_fps

    def _update_pipes(self, dt):
        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed * dt
        self.pipes = [p for p in self.pipes if p["x"] > -self.pipe_width]

        if not self.pipes or self.pipes[-1]["x"] < self.width - self.pipe_spacing:
            gap_y = random.randint(140, self.height - 140)
            self.pipes.append({"x": self.width + 40, "gap_y": gap_y})

    def _check_collisions(self):
        bird_rect = pygame.Rect(self.bird_x - 18, self.bird_y - 14, 36, 28)
        if self.bird_y < 0 or self.bird_y > self.height:
            self._trigger_game_over()
            return
        for pipe in self.pipes:
            top_height = pipe["gap_y"] - self.pipe_gap / 2
            bottom_y = pipe["gap_y"] + self.pipe_gap / 2
            top_rect = pygame.Rect(pipe["x"], 0, self.pipe_width, top_height)
            bottom_rect = pygame.Rect(
                pipe["x"], bottom_y, self.pipe_width, self.height - bottom_y
            )
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                self._trigger_game_over()
                return

    def _trigger_game_over(self):
        if not self.game_over:
            self.game_over = True
            self.game_over_time = time.time()

    def draw(self, arm_level):
        self.screen.fill((60, 180, 210))
        self._draw_pipes()
        self._draw_bird()

        info = self.font.render(f"Arm lift: {arm_level:.2f}", True, (10, 10, 10))
        self.screen.blit(info, (10, 10))

        if self.game_over:
            text = self.font.render("Game Over - resetting...", True, (220, 20, 20))
            self.screen.blit(
                text,
                (self.width / 2 - text.get_width() / 2, self.height / 2 - 20),
            )

        pygame.display.flip()

    def _draw_pipes(self):
        for pipe in self.pipes:
            top_height = pipe["gap_y"] - self.pipe_gap / 2
            bottom_y = pipe["gap_y"] + self.pipe_gap / 2
            pygame.draw.rect(
                self.screen, (20, 150, 20), (pipe["x"], 0, self.pipe_width, top_height)
            )
            pygame.draw.rect(
                self.screen,
                (20, 150, 20),
                (pipe["x"], bottom_y, self.pipe_width, self.height - bottom_y),
            )

    def _draw_bird(self):
        if self.bird_frames:
            frame = self.bird_frames[self.bird_frame_index]
            rect = frame.get_rect(center=(self.bird_x, self.bird_y))
            self.screen.blit(frame, rect.topleft)
        else:
            pygame.draw.ellipse(
                self.screen, (250, 230, 50), (self.bird_x - 18, self.bird_y - 14, 36, 28)
            )

    def _load_bird_frames(self):
        frames = []
        base_dir = os.path.dirname(__file__)
        frame_files = ["Frame-1.png", "frame-2.png", "frame-3.png", "frame-4.png"]
        for filename in frame_files:
            path = os.path.join(base_dir, "birdCharacter", "PNG", filename)
            if not os.path.exists(path):
                continue
            image = pygame.image.load(path).convert_alpha()
            image = pygame.transform.smoothscale(image, (48, 36))
            frames.append(image)
        return frames


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    pose = PoseController(smooth_alpha=0.3)
    game = FlappyGame()
    flap_threshold = 0.8
    flap_reset = 0.6
    flap_armed = True
    running = True

    while running:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        annotated, arm_level, detected = pose.process(frame)
        flap = False
        if not detected:
            flap_armed = True
        elif flap_armed and arm_level >= flap_threshold:
            flap = True
            flap_armed = False
        elif not flap_armed and arm_level <= flap_reset:
            flap_armed = True
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f} detected: {detected} flap: {flap}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.tick(arm_level if detected else 0.0, flap)
        game.draw(arm_level if detected else 0.0)

        cv2.imshow("Pose Control", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
