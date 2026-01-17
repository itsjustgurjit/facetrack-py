import cv2
import mediapipe as mp
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
import os
import psutil
from tqdm import tqdm
import time

class VideoCropper:
    def __init__(self, input_path, output_path, model_path="pose_landmarker_heavy.task"):
        self.input_path = input_path
        self.output_path = output_path
        
        # Initialize MediaPipe Pose Landmarker (Tasks API)
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create the landmarker with the model file
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def get_subject_center(self, frame):
        """Detects the subject center in the frame using MediaPipe Tasks API."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        detection_result = self.landmarker.detect(mp_image)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0] # First person
            
            # Map landmarks specifically
            # Note: tasks API landmarks are object with x,y,z,visibility,presence
            # Indexes: Nose=0, Left Hip=23, Right Hip=24
            nose = landmarks[0]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate a robust center (weighted towards the head but grounded)
            # x is normalized [0, 1]
            center_x = (nose.x + (left_hip.x + right_hip.x) / 2) / 2 * w
            return center_x
        
        return None

    def get_resource_usage(self):
        """Returns current CPU and RAM usage."""
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        ram_used_gb = psutil.virtual_memory().used / (1024 ** 3)
        return f"CPU: {cpu_percent}% | RAM: {ram_used_gb:.2f}GB"

    def smooth_centers(self, centers, alpha=0.1):
        """Applies Exponential Moving Average (EMA) to smooth center points."""
        smoothed = []
        last_val = centers[0] if centers else 0
        
        for val in centers:
            if val is None:
                current_val = last_val
            else:
                current_val = val
                
            # EMA formula
            smoothed_val = alpha * current_val + (1 - alpha) * last_val
            smoothed.append(smoothed_val)
            last_val = smoothed_val
            
        return smoothed

    def process(self):
        print(f"Opening video: {self.input_path}")
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_height = height
        target_width = int(target_height * 9 / 16)
        
        print(f"Original: {width}x{height}, Target: {target_width}x{target_height}")

        # Pass 1: Analyze frames for tracking
        print("Pass 1: Analyzing frames for tracking...")
        centers = []
        frame_count = 0
        
        pbar = tqdm(total=total_frames, unit="frame", desc="Tracking")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            center_x = self.get_subject_center(frame)
            centers.append(center_x)
            
            frame_count += 1
            if frame_count % 10 == 0:
                pbar.set_postfix_str(self.get_resource_usage())
            pbar.update(1)

        pbar.close()
        cap.release()
        
        if not centers:
            print("No detection. Exiting.")
            return

        # Fill missing detections with nearest valid value
        # Simple forward/backward fill
        clean_centers = []
        last_valid = width // 2 # Default to center
        
        # First pass to fill forward
        for c in centers:
            if c is not None:
                last_valid = c
            clean_centers.append(last_valid)
            
        # Smoothing
        smoothed_centers = self.smooth_centers(clean_centers, alpha=0.05) # Lower alpha = smoother

        # Pass 2: Crop and Save
        print("Pass 2: Cropping and saving video...")
        cap = cv2.VideoCapture(self.input_path)
        
        # Temp output for video only
        temp_video_path = "temp_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))

        pbar = tqdm(total=total_frames, unit="frame", desc="Cropping")

        for i in range(len(smoothed_centers)):
            ret, frame = cap.read()
            if not ret:
                break
            
            center_x = smoothed_centers[i]
            
            # Calculate crop coordinates
            start_x = int(center_x - target_width // 2)
            
            # Boundary checks
            if start_x < 0:
                start_x = 0
            elif start_x + target_width > width:
                start_x = width - target_width
            
            crop_img = frame[0:target_height, start_x:start_x+target_width]
            out.write(crop_img)

            if i % 10 == 0:
                 pbar.set_postfix_str(self.get_resource_usage())
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()
        self.landmarker.close()

        # Add Audio
        print("Adding audio...")
        try:
            video_clip = VideoFileClip(temp_video_path)
            original_clip = VideoFileClip(self.input_path)
            
            # Check if original has audio
            if original_clip.audio:
                final_clip = video_clip.with_audio(original_clip.audio)
                final_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')
            else:
                 # No audio, just rename temp or write without audio
                 print("Video has no audio. Saving silent video.")
                 video_clip.write_videofile(self.output_path, codec='libx264')

            video_clip.close()
            original_clip.close()
            
            # Cleanup
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                
            print(f"Done! Saved to {self.output_path}")
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            print(f"Silent video saved as {temp_video_path}")

if __name__ == "__main__":
    # You can change these paths as needed, or make them command line args
    INPUT_FILE = "input.mp4" 
    OUTPUT_FILE = "output_vertical.mp4"
    MODEL_FILE = "pose_landmarker_heavy.task"
    
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found. Please run download_model.py")
        exit(1)

    if os.path.exists(INPUT_FILE):
        cropper = VideoCropper(INPUT_FILE, OUTPUT_FILE, MODEL_FILE)
        cropper.process()
    else:
        print(f"Please place an input video at {INPUT_FILE} or update the path in main.py")
