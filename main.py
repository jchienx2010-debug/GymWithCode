import math
import os
import random
import time
import sys

# Fix SDL2 library conflict on macOS between pygame and cv2
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# Set SDL to use dummy video driver for OpenCV if available
# This prevents conflicts between pygame and cv2 on macOS
if sys.platform == 'darwin':  # macOS
    # Tell OpenCV to avoid competing with pygame for SDL
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Initialize pygame first to claim SDL2
import pygame
pygame.init()

import cv2
import numpy as np
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
        # pygame already initialized at module level
        self.width = width
        self.height = height
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Pose Flappy Bird")
        self.clock = pygame.time.Clock()
        self.bird_frame_index = 0
        self.bird_anim_time = 0.0
        self.bird_anim_fps = 10.0
        self.flap_cooldown = 0.22
        self.elapsed_time = 0.0  # Initialize before _update_scale
        self._update_scale()  # This will also load bird frames
        self.reset()

    def _update_scale(self):
        """Update scaled values based on current screen dimensions."""
        # Scale relative to 640x480 base resolution
        scale_x = self.width / 640.0
        scale_y = self.height / 480.0
        self.scale = min(scale_x, scale_y)
        
        # Update font size based on scale
        font_size = max(int(24 * self.scale), 12)
        self.font = pygame.font.SysFont("arial", font_size)
        
        # Scaled game parameters
        self.bird_x_base = 120
        self.pipe_gap_base = 160
        self.pipe_width_base = 140  # Increased from 70 to make pillars thicker
        self.pipe_spacing_base_initial = 480  # Start wider apart
        self.pipe_spacing_base_min = 260  # End closer together
        self.pipe_speed_base = 120.0
        self.gravity_base = 470.0
        self.flap_strength_base = 300.0
        
        self.bird_x = self.bird_x_base * scale_x
        self.pipe_gap = self.pipe_gap_base * scale_y
        self.pipe_width = self.pipe_width_base * scale_x
        self.pipe_speed = self.pipe_speed_base * scale_x
        self.gravity = self.gravity_base * scale_y
        self.flap_strength = self.flap_strength_base * scale_y
        
        # Initialize pillar sprite and hitbox defaults before _update_pipe_spacing
        self.pillar_sprite = None  # Initialize early to avoid AttributeError
        self.pillar_visual_x_ratio = 0.0
        self.pillar_visual_y_ratio = 0.0
        self.pillar_visual_width_ratio = 1.0
        self.pillar_visual_height_ratio = 1.0
        
        # Pipe spacing is updated dynamically based on elapsed time
        self._update_pipe_spacing()
        
        # Scaled bird size
        self.bird_width = int(48 * scale_x)
        self.bird_height = int(36 * scale_y)
        self.bird_half_width = self.bird_width // 2
        self.bird_half_height = self.bird_height // 2
        
        # Reload bird frames at new size
        self.bird_frames = self._load_bird_frames()
        
        # Load pillar sprite (will update hitbox ratios if sprite loads successfully)
        self.pillar_sprite = self._load_pillar_sprite()

    def reset(self):
        self.bird_y = self.height / 2
        self.bird_vel = 0.0
        self.bird_frame_index = 0
        self.bird_anim_time = 0.0
        self.last_flap_time = 0.0
        self.pipes = []
        self.last_pipe_x = self.width
        self.game_over = False
        self.game_over_time = 0.0
        self.score = 0
        self.start_time = time.time()
        self.elapsed_time = 0.0
        self._update_pipe_spacing()

    def _update_pipe_spacing(self):
        """Update pipe spacing based on elapsed time - starts wide and gets narrower."""
        # Decrease spacing from initial to minimum over 60 seconds
        progress = min(self.elapsed_time / 60.0, 1.0)
        spacing_base = self.pipe_spacing_base_initial - (self.pipe_spacing_base_initial - self.pipe_spacing_base_min) * progress
        scale_x = self.width / 640.0
        self.pipe_spacing = spacing_base * scale_x
        
        # Update hitbox width to scale with spacing
        # When pipes are far apart (progress=0), hitbox is 100% width
        # When pipes are close (progress=1), hitbox is 60% width
        hitbox_scale = 1.0 - (progress * 0.4)  # Ranges from 1.0 to 0.6
        
        if self.pillar_sprite and hasattr(self, 'pillar_visual_width_ratio'):
            # Scale the hitbox based on spacing progress
            base_hitbox = self.pipe_width * self.pillar_visual_width_ratio
            self.pillar_hitbox_width = base_hitbox * hitbox_scale
        else:
            # Fallback if no sprite
            self.pillar_hitbox_width = self.pipe_width * hitbox_scale

    def tick(self, arm_level, flap):
        dt = self.clock.tick(60) / 1000.0
        if self.game_over:
            if time.time() - self.game_over_time > 2.0:
                self.reset()
            return dt

        # Update elapsed time
        self.elapsed_time = time.time() - self.start_time
        self._update_pipe_spacing()  # Update spacing based on time

        now = time.time()
        if flap and (now - self.last_flap_time) >= self.flap_cooldown:
            self.bird_vel = -self.flap_strength
            self.last_flap_time = now
        self.bird_vel += self.gravity * dt
        self.bird_y += self.bird_vel * dt
        self._update_animation(dt)

        self._update_pipes(dt)
        self._update_score()
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
            # Scale the margin based on screen height
            margin = int(140 * (self.height / 480.0))
            gap_y = random.randint(margin, self.height - margin)
            self.pipes.append({"x": self.width + 40, "gap_y": gap_y, "scored": False})

    def _update_score(self):
        """Check if bird has passed any pipes and increment score."""
        for pipe in self.pipes:
            if not pipe["scored"] and pipe["x"] + self.pipe_width < self.bird_x:
                pipe["scored"] = True
                self.score += 1

    def _check_collisions(self):
        bird_rect = pygame.Rect(
            self.bird_x - self.bird_half_width, 
            self.bird_y - self.bird_half_height, 
            self.bird_width, 
            self.bird_height
        )
        
        # Check ceiling or floor collision - game over
        if self.bird_y < 0 or self.bird_y > self.height:
            self._trigger_game_over()
            return
            
        for pipe in self.pipes:
            top_height = pipe["gap_y"] - self.pipe_gap / 2
            bottom_y = pipe["gap_y"] + self.pipe_gap / 2
            
            if self.pillar_sprite:
                # Calculate hitboxes using stored ratios from original sprite
                # Top pillar hitbox - account for vertical flip transformation
                if top_height > 0:
                    hitbox_x = pipe["x"] + (self.pipe_width * self.pillar_visual_x_ratio)
                    # When flipped, visual y position transforms: new_y = height * (1 - y_ratio - height_ratio)
                    hitbox_y = top_height * (1.0 - self.pillar_visual_y_ratio - self.pillar_visual_height_ratio)
                    hitbox_width = self.pipe_width * self.pillar_visual_width_ratio
                    hitbox_height = top_height * self.pillar_visual_height_ratio
                    top_rect = pygame.Rect(hitbox_x, hitbox_y, hitbox_width, hitbox_height)
                else:
                    top_rect = None
                
                # Bottom pillar hitbox
                bottom_height = self.height - bottom_y
                if bottom_height > 0:
                    hitbox_x = pipe["x"] + (self.pipe_width * self.pillar_visual_x_ratio)
                    hitbox_y = bottom_y + (bottom_height * self.pillar_visual_y_ratio)
                    hitbox_width = self.pipe_width * self.pillar_visual_width_ratio
                    hitbox_height = bottom_height * self.pillar_visual_height_ratio
                    bottom_rect = pygame.Rect(hitbox_x, hitbox_y, hitbox_width, hitbox_height)
                else:
                    bottom_rect = None
            else:
                # Fallback to simple rectangles
                top_rect = pygame.Rect(pipe["x"], 0, self.pipe_width, top_height)
                bottom_rect = pygame.Rect(pipe["x"], bottom_y, self.pipe_width, self.height - bottom_y)
            
            # Check collisions
            if top_rect and bird_rect.colliderect(top_rect):
                self._trigger_game_over()
                return
            if bottom_rect and bird_rect.colliderect(bottom_rect):
                self._trigger_game_over()
                return

    def _trigger_game_over(self):
        if not self.game_over:
            self.game_over = True
            self.game_over_time = time.time()

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            # Use borderless window at screen resolution instead of exclusive fullscreen
            # This allows window switching (Alt-Tab/Cmd-Tab) and multi-monitor support
            info = pygame.display.Info()
            self.width = info.current_w
            self.height = info.current_h
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
        else:
            self.width = 640
            self.height = 480
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        
        # Update scaled values for new dimensions
        self._update_scale()
        # Adjust bird position for new screen size
        if hasattr(self, 'bird_y'):
            # Keep bird at same relative position
            self.bird_y = min(self.bird_y, self.height - 50)

    def draw(self, arm_level, detected):
        self.screen.fill((60, 180, 210))
        self._draw_pipes()
        self._draw_bird()

        # Calculate spacing based on font size
        line_height = self.font.get_height() + 5
        
        # Draw score (top left)
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Draw timer (top right)
        elapsed_time = time.time() - self.start_time if not self.game_over else self.game_over_time - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        timer_text = self.font.render(f"Time: {minutes}:{seconds:02d}", True, (255, 255, 255))
        self.screen.blit(timer_text, (self.width - timer_text.get_width() - 10, 10))
        
        # Draw camera positioning guidance (below score, with proper spacing)
        if detected:
            guidance = "Camera: Good! | Controls: Lower arms to flap"
            color = (0, 255, 0)  # Green
        else:
            guidance = "Camera: Adjust position"
            color = (255, 100, 0)  # Orange
        guidance_text = self.font.render(guidance, True, color)
        self.screen.blit(guidance_text, (10, 10 + line_height))

        if self.game_over:
            game_over_text = self.font.render("Game Over - resetting...", True, (220, 20, 20))
            final_score_text = self.font.render(f"Final Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(
                game_over_text,
                (self.width / 2 - game_over_text.get_width() / 2, self.height / 2 - 40),
            )
            self.screen.blit(
                final_score_text,
                (self.width / 2 - final_score_text.get_width() / 2, self.height / 2),
            )

        pygame.display.flip()

    def _draw_pipes(self):
        for pipe in self.pipes:
            top_height = pipe["gap_y"] - self.pipe_gap / 2
            bottom_y = pipe["gap_y"] + self.pipe_gap / 2
            
            if self.pillar_sprite:
                # Draw top pillar - flip vertically and stretch to fill from top to gap
                if top_height > 0:
                    top_sprite = pygame.transform.flip(self.pillar_sprite, False, True)
                    stretched_top = pygame.transform.scale(top_sprite, (int(self.pipe_width), int(top_height)))
                    self.screen.blit(stretched_top, (pipe["x"], 0))
                    
                    # Calculate hitbox - account for vertical flip transformation
                    # When flipped, visual content that was at y_offset is now at (1 - y_offset - height_ratio)
                    hitbox_x = pipe["x"] + (self.pipe_width * self.pillar_visual_x_ratio)
                    hitbox_y = top_height * (1.0 - self.pillar_visual_y_ratio - self.pillar_visual_height_ratio)
                    hitbox_width = self.pipe_width * self.pillar_visual_width_ratio
                    hitbox_height = top_height * self.pillar_visual_height_ratio
                    top_rect = pygame.Rect(hitbox_x, hitbox_y, hitbox_width, hitbox_height)
                else:
                    top_rect = None
                
                # Draw bottom pillar - stretch to fill from gap to bottom
                bottom_height = self.height - bottom_y
                if bottom_height > 0:
                    stretched_bottom = pygame.transform.scale(self.pillar_sprite, (int(self.pipe_width), int(bottom_height)))
                    self.screen.blit(stretched_bottom, (pipe["x"], bottom_y))
                    
                    # Calculate hitbox using stored ratios from original sprite
                    hitbox_x = pipe["x"] + (self.pipe_width * self.pillar_visual_x_ratio)
                    hitbox_y = bottom_y + (bottom_height * self.pillar_visual_y_ratio)
                    hitbox_width = self.pipe_width * self.pillar_visual_width_ratio
                    hitbox_height = bottom_height * self.pillar_visual_height_ratio
                    bottom_rect = pygame.Rect(hitbox_x, hitbox_y, hitbox_width, hitbox_height)
                else:
                    bottom_rect = None
            else:
                # Fallback to rectangles if no sprite
                pygame.draw.rect(self.screen, (20, 150, 20), (pipe["x"], 0, self.pipe_width, top_height))
                pygame.draw.rect(self.screen, (20, 150, 20), (pipe["x"], bottom_y, self.pipe_width, self.height - bottom_y))
                top_rect = pygame.Rect(pipe["x"], 0, self.pipe_width, top_height)
                bottom_rect = pygame.Rect(pipe["x"], bottom_y, self.pipe_width, self.height - bottom_y)
            
            # Draw hitbox outlines using sprite-specific dimensions
            if top_rect:
                pygame.draw.rect(self.screen, (255, 0, 0), top_rect, 2)  # Red outline
            if bottom_rect:
                pygame.draw.rect(self.screen, (255, 0, 0), bottom_rect, 2)  # Red outline

    def _draw_bird(self):
        # Draw bird sprite or ellipse
        if self.bird_frames:
            frame = self.bird_frames[self.bird_frame_index]
            rect = frame.get_rect(center=(self.bird_x, self.bird_y))
            self.screen.blit(frame, rect.topleft)
        else:
            pygame.draw.ellipse(
                self.screen, 
                (250, 230, 50), 
                (self.bird_x - self.bird_half_width, 
                 self.bird_y - self.bird_half_height, 
                 self.bird_width, 
                 self.bird_height)
            )
        
        # Draw bird hitbox outline
        bird_hitbox = pygame.Rect(
            self.bird_x - self.bird_half_width, 
            self.bird_y - self.bird_half_height, 
            self.bird_width, 
            self.bird_height
        )
        pygame.draw.rect(self.screen, (255, 0, 0), bird_hitbox, 2)  # Red outline

    def _load_bird_frames(self):
        frames = []
        base_dir = os.path.dirname(__file__)
        bird_dir = os.path.join(base_dir, "birdCharacter", "PNG")
        
        # Load all PNG files from the bird directory
        if os.path.exists(bird_dir):
            png_files = [f for f in os.listdir(bird_dir) if f.endswith('.png')]
            # Sort files alphabetically to maintain consistent animation order
            png_files.sort()
            
            for filename in png_files:
                path = os.path.join(bird_dir, filename)
                try:
                    image = pygame.image.load(path).convert_alpha()
                    # Use scaled bird dimensions
                    image = pygame.transform.smoothscale(image, (self.bird_width, self.bird_height))
                    frames.append(image)
                except Exception as e:
                    print(f"Could not load bird sprite {filename}: {e}")
        
        return frames

    def _load_pillar_sprite(self):
        """Load the pillar sprite and scale it to match pipe width."""
        base_dir = os.path.dirname(__file__)
        sprite_path = os.path.join(base_dir, "pillars", "PNG", "pixil-frame-0.png")
        
        if not os.path.exists(sprite_path):
            return None
        
        try:
            sprite = pygame.image.load(sprite_path).convert_alpha()
            # Get original dimensions
            original_width = sprite.get_width()
            original_height = sprite.get_height()
            
            # Calculate actual visual content by finding non-transparent pixels
            # Get the bounding rect of non-transparent pixels
            mask = pygame.mask.from_surface(sprite)
            bounding_rect = mask.get_bounding_rects()
            
            if bounding_rect:
                # Use the first bounding rect to determine actual visual bounds
                actual_rect = bounding_rect[0]
                # Store all ratios for x, y, width, height relative to sprite dimensions
                self.pillar_visual_x_ratio = actual_rect.x / original_width
                self.pillar_visual_y_ratio = actual_rect.y / original_height
                self.pillar_visual_width_ratio = actual_rect.width / original_width
                self.pillar_visual_height_ratio = actual_rect.height / original_height
            else:
                # Fallback if no bounding rect found
                self.pillar_visual_x_ratio = 0.0
                self.pillar_visual_y_ratio = 0.0
                self.pillar_visual_width_ratio = 1.0
                self.pillar_visual_height_ratio = 1.0
            
            # Scale sprite to match pipe width using nearest-neighbor to preserve pixelation
            scale_factor = self.pipe_width / original_width
            new_height = int(original_height * scale_factor)
            # Use scale instead of smoothscale to preserve pixelated look
            scaled_sprite = pygame.transform.scale(sprite, (int(self.pipe_width), new_height))
            
            return scaled_sprite
        except Exception as e:
            print(f"Error loading pillar sprite: {e}")
            # Fallback values if loading fails
            self.pillar_visual_x_ratio = 0.0
            self.pillar_visual_y_ratio = 0.0
            self.pillar_visual_width_ratio = 1.0
            self.pillar_visual_height_ratio = 1.0
            return None


def calibrate_arm_positions(cap, pose):
    """
    Calibration phase to determine personalized arm thresholds.
    Returns (flap_threshold, flap_reset) based on user's arm range.
    """
    print("\n=== CALIBRATION PHASE ===")
    print("This will help personalize the controls to your arm movement.\n")
    
    cv2.namedWindow("Pose Control", cv2.WINDOW_NORMAL)
    
    # Step 1: Wait for arms to reach T-pose position, then calibrate
    print("Step 1: ARMS STRAIGHT OUT TO THE SIDES (T-pose)")
    print("Arms should be perpendicular to your body (90 degrees)")
    print("Waiting for you to extend your arms...")
    
    # Wait until arms are detected in proper T-pose (arm_level > 0.85 = ~80 degrees)
    arms_detected_up = False
    while not arms_detected_up:
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        annotated, arm_level, detected = pose.process(frame)
        
        if detected and arm_level > 0.85:
            arms_detected_up = True
        
        cv2.putText(
            annotated,
            "T-POSE: ARMS STRAIGHT OUT TO SIDES",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f} - Need > 0.85",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        
        cv2.imshow("Pose Control", annotated)
        cv2.waitKey(1)
    
    print("✓ Arms detected up! Hold this T-pose steady for 3 seconds...")
    
    arms_up_samples = []
    start_time = time.time()
    calibration_duration = 3.0
    
    while time.time() - start_time < calibration_duration:
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        annotated, arm_level, detected = pose.process(frame)
        
        if detected:
            arms_up_samples.append(arm_level)
        
        # Display countdown
        remaining = calibration_duration - (time.time() - start_time)
        cv2.putText(
            annotated,
            f"HOLD T-POSE STEADY - {remaining:.1f}s",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        
        cv2.imshow("Pose Control", annotated)
        cv2.waitKey(1)
    
    arms_up_level = sum(arms_up_samples) / len(arms_up_samples) if arms_up_samples else 0.8
    print(f"✓ T-pose position recorded: {arms_up_level:.2f}")
    
    # Step 2: Wait for arms to go DOWN, then calibrate
    print("\nStep 2: LOWER YOUR ARMS DOWN TO YOUR SIDES")
    print("Waiting for you to lower your arms...")
    
    # Wait until arms are detected as down (arm_level < 0.5)
    arms_detected_down = False
    while not arms_detected_down:
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        annotated, arm_level, detected = pose.process(frame)
        
        if detected and arm_level < 0.5:
            arms_detected_down = True
        
        cv2.putText(
            annotated,
            "LOWER YOUR ARMS DOWN",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f} - Need < 0.50",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        
        cv2.imshow("Pose Control", annotated)
        cv2.waitKey(1)
    
    print("✓ Arms detected down! Hold this position steady for 3 seconds...")
    
    arms_down_samples = []
    start_time = time.time()
    
    while time.time() - start_time < calibration_duration:
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        annotated, arm_level, detected = pose.process(frame)
        
        if detected:
            arms_down_samples.append(arm_level)
        
        # Display countdown
        remaining = calibration_duration - (time.time() - start_time)
        cv2.putText(
            annotated,
            f"HOLD ARMS DOWN - {remaining:.1f}s",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        
        cv2.imshow("Pose Control", annotated)
        cv2.waitKey(1)
    
    arms_down_level = sum(arms_down_samples) / len(arms_down_samples) if arms_down_samples else 0.2
    print(f"✓ Arms down position recorded: {arms_down_level:.2f}")
    
    # Calculate thresholds based on calibrated range
    # Flap threshold: 30% up from the down position
    # Reset threshold: 70% up from the down position
    range_span = arms_up_level - arms_down_level
    flap_threshold = arms_down_level + (range_span * 0.30)
    flap_reset = arms_down_level + (range_span * 0.70)
    
    print(f"\n✓ Calibration complete!")
    print(f"  Flap trigger threshold: {flap_threshold:.2f} (lower arms to this level to flap)")
    print(f"  Reset threshold: {flap_reset:.2f} (raise arms to this level to re-arm)")
    print("\nNow return to T-pose (arms straight out to sides) to start the game...")
    
    # Step 3: Wait for player to return to T-pose before starting
    ready_to_start = False
    while not ready_to_start:
        ok, frame = cap.read()
        if not ok:
            continue
        
        frame = cv2.flip(frame, 1)
        annotated, arm_level, detected = pose.process(frame)
        
        if detected and arm_level >= flap_reset:
            ready_to_start = True
        
        cv2.putText(
            annotated,
            "T-POSE: ARMS OUT TO SIDES TO START",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
        )
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f} - Need >= {flap_reset:.2f}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Thresholds - Flap: {flap_threshold:.2f} | Reset: {flap_reset:.2f}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        
        cv2.imshow("Pose Control", annotated)
        cv2.waitKey(1)
    
    print("✓ T-pose detected! Starting game in 2 seconds...\n")
    
    # Show ready message
    start_time = time.time()
    while time.time() - start_time < 2.0:
        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            annotated, arm_level, _ = pose.process(frame)
            remaining = 2.0 - (time.time() - start_time)
            cv2.putText(
                annotated,
                f"READY! Starting in {remaining:.1f}s",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                annotated,
                "Lower arms to flap!",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Pose Control", annotated)
            cv2.waitKey(1)
    
    return flap_threshold, flap_reset


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    pose = PoseController(smooth_alpha=0.3)
    
    # Run calibration phase
    flap_threshold, flap_reset = calibrate_arm_positions(cap, pose)
    
    # Create game after calibration
    game = FlappyGame()
    
    # Position-based controls - drop from T-pose and return to T-pose to flap
    flap_armed = True  # Start armed (in T-pose)
    running = True
    camera_fullscreen = False

    while running:
        ok, frame = cap.read()
        if not ok:
            # Sometimes the first frame fails
            continue
            
        frame = cv2.flip(frame, 1)
        
        annotated, arm_level, detected = pose.process(frame)
        
        flap = False
        # Only update flap state when pose is actually detected
        if detected:
            if arm_level < flap_threshold:
                # Arms are down - disarm (waiting to return to T-pose)
                flap_armed = False
            elif not flap_armed and arm_level >= flap_reset:
                # Returning to T-pose from down position - FLAP!
                flap = True
                flap_armed = True
            # If armed and in T-pose range (between threshold and reset), stay armed
            
        cv2.putText(
            annotated,
            f"arm_level: {arm_level:.2f} | armed: {flap_armed} | flap: {flap}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if detected else (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Drop arms & return to T-pose to flap | T-pose: >{flap_reset:.2f} | Down: <{flap_threshold:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    game.toggle_fullscreen()
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                game.width = event.w
                game.height = event.h
                game.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                game._update_scale()
                # Adjust bird position to stay within bounds
                if hasattr(game, 'bird_y'):
                    game.bird_y = min(game.bird_y, game.height - 50)

        game.tick(arm_level if detected else 0.0, flap)
        game.draw(arm_level if detected else 0.0, detected)

        cv2.imshow("Pose Control", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            print("ESC key pressed - exiting")
            running = False
        elif key == ord('f') or key == ord('F'):  # F key for camera fullscreen
            camera_fullscreen = not camera_fullscreen
            if camera_fullscreen:
                cv2.setWindowProperty("Pose Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("Pose Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()