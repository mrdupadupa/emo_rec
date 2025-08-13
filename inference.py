
import time
import os
import gc
import numpy as np
import pygame
import pygame.midi
import cv2
import random
import threading
from deepface import DeepFace
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# ===== Configuration =====
# Path to emotion samples
SAMPLES_DIR = "/home/andrei/Work/emotion_recognition/emotion_samples"

# Emotion detection settings
DETECTION_INTERVAL = 3  # seconds
TRANSITION_TIME = 1.5  # seconds for crossfade

# Face weighting settings
CENTER_WEIGHT = 0.7  # Faces in center get higher weight
SIZE_WEIGHT = 0.3    # Larger faces get higher weight


# ===== Music Player Class =====
class EmotionMusicPlayer:
    """Handles playback and crossfading of emotion-based music."""
    
    def __init__(self):
        """Initialize the player."""
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        pygame.midi.init()
        
        self.current_emotion = None
        self.current_sample = None
        self.next_emotion = None
        self.next_sample = None
        self.is_transitioning = False
        self.transition_thread = None
        
        # Pre-load sample paths
        self.sample_paths = self._load_sample_paths()
        
    def _load_sample_paths(self):
        """Load all sample paths from the samples directory."""
        sample_paths = {}
        
        for emotion in os.listdir(SAMPLES_DIR):
            emotion_dir = os.path.join(SAMPLES_DIR, emotion)
            if os.path.isdir(emotion_dir):
                sample_paths[emotion] = []
                for file in os.listdir(emotion_dir):
                    if file.endswith('.mid'):
                        sample_paths[emotion].append(os.path.join(emotion_dir, file))
        
        return sample_paths
    
    def get_sample_for_emotion(self, emotion):
        """Get a random sample for the given emotion."""
        if emotion in self.sample_paths and self.sample_paths[emotion]:
            return random.choice(self.sample_paths[emotion])
        
        # If no samples for this emotion, try a fallback
        fallbacks = {
            'happy_sad': ['happy', 'sad'],
            'happy_angry': ['happy', 'angry'],
            'sad_fear': ['sad', 'fear'],
            'surprise_happy': ['surprise', 'happy']
            # Add more fallbacks as needed
        }
        
        if emotion in fallbacks:
            for fallback in fallbacks[emotion]:
                if fallback in self.sample_paths and self.sample_paths[fallback]:
                    return random.choice(self.sample_paths[fallback])
        
        # Last resort, use neutral
        if 'neutral' in self.sample_paths and self.sample_paths['neutral']:
            return random.choice(self.sample_paths['neutral'])
        
        return None
    
    def play_emotion(self, emotion):
        """Play music for a specific emotion."""
        if emotion == self.current_emotion and pygame.mixer.music.get_busy():
            return  # Already playing this emotion
        
        sample_path = self.get_sample_for_emotion(emotion)
        if not sample_path:
            print(f"No sample found for emotion: {emotion}")
            return
        
        if pygame.mixer.music.get_busy():
            # Start transition to new emotion
            self.next_emotion = emotion
            self.next_sample = sample_path
            
            if not self.is_transitioning:
                self.is_transitioning = True
                self.transition_thread = threading.Thread(target=self._transition_to_next)
                self.transition_thread.daemon = True
                self.transition_thread.start()
        else:
            # Start playing immediately
            try:
                pygame.mixer.music.load(sample_path)
                pygame.mixer.music.play()
                self.current_emotion = emotion
                self.current_sample = sample_path
                print(f"Now playing: {emotion} - {os.path.basename(sample_path)}")
            except Exception as e:
                print(f"Error playing sample {sample_path}: {e}")
    
    def _transition_to_next(self):
        """Handle smooth transition between emotions."""
        try:
            # Fade out current music
            pygame.mixer.music.fadeout(int(TRANSITION_TIME * 1000))
            
            # Wait for fadeout to complete
            time.sleep(TRANSITION_TIME)
            
            # Load and play next sample
            pygame.mixer.music.load(self.next_sample)
            pygame.mixer.music.play()
            
            # Update current emotion
            self.current_emotion = self.next_emotion
            self.current_sample = self.next_sample
            self.next_emotion = None
            self.next_sample = None
            
            print(f"Transitioned to: {self.current_emotion} - {os.path.basename(self.current_sample)}")
        except Exception as e:
            print(f"Error during transition: {e}")
        finally:
            self.is_transitioning = False

# ===== Face Processing Functions =====
def calculate_face_weights(faces, frame_width, frame_height):
    """Calculate weights for each face based on size and position."""
    # Check if faces is empty - handle both tuple and numpy array cases
    if isinstance(faces, tuple) or len(faces) == 0:
        return []
    

    weights = []
    for (x, y, w, h) in faces:
        # Calculate face size (area)
        size = w * h
        
        # Calculate distance from center of frame
        center_x = frame_width / 2
        center_y = frame_height / 2
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        
        # Normalize distance (0 = center, 1 = edge)
        max_distance = np.sqrt((center_x)**2 + (center_y)**2)
        distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
        normalized_distance = distance / max_distance
        
        # Center factor (higher for faces closer to center)
        center_factor = 1 - normalized_distance
        
        # Calculate combined weight
        # Normalize size by the frame area
        normalized_size = size / (frame_width * frame_height)
        weight = CENTER_WEIGHT * center_factor + SIZE_WEIGHT * normalized_size
        
        weights.append(weight)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight > 0:
        return [w / total_weight for w in weights]
    return [1.0 / len(weights)] * len(weights)  # Equal weights if total is 0

def blend_emotions(emotions, weights):
    """Blend emotions based on weights."""
    if not emotions:
        return "neutral"
    
    # Count weighted occurrences of each emotion
    emotion_weights = {}
    for emotion, weight in zip(emotions, weights):
        emotion_weights[emotion] = emotion_weights.get(emotion, 0) + weight
    
    # Get the emotion with the highest weight
    primary_emotion = max(emotion_weights, key=emotion_weights.get)
    
    # If there's a clear dominant emotion (weight > 0.6), use it
    if emotion_weights[primary_emotion] > 0.6:
        return primary_emotion
    
    # Otherwise, find the top two emotions and create a blended name
    sorted_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_emotions) >= 2:
        top_emotion, top_weight = sorted_emotions[0]
        second_emotion, second_weight = sorted_emotions[1]
        
        # If the weights are similar, create a blended emotion name
        if second_weight > 0.3:
            blended_name = f"{top_emotion}_{second_emotion}"
            # Check if we have samples for this blend, otherwise use primary
            return blended_name if blended_name in player.sample_paths else top_emotion
    
    # Default to primary emotion
    return primary_emotion

# ===== Main Program =====
#configuration settings
WINDOW_NAME = 'Emotion-Based Music Generation'
WINDOW_WIDTH = 1920  # Larger window width
WINDOW_HEIGHT = 1280  # Larger window height
BLINK_DURATION = 0.5  # seconds to show the detection indicator
DETECTION_COLOR = (0, 0, 255)  # Red color for detection indicator
NORMAL_COLOR = (0, 255, 0)  # Green color for normal state

def main():
    # Initialize browser for visualization (optional)
    geckodriver_path = "/snap/bin/geckodriver"
    driver_service = webdriver.FirefoxService(executable_path=geckodriver_path)
    driver = webdriver.Firefox(service=driver_service)
    driver.get("http://127.0.0.1:9966/")
    seed_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize emotion music player
    global player
    player = EmotionMusicPlayer()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Try to set higher camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    
    # Set up display window properties
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Variables for tracking
    last_detection_time = 0
    detected_faces = []
    emotions = []
    current_blended_emotion = "neutral"
    
    # Variables for visual indicator
    is_detecting = False
    detection_start_time = 0
    frame_border_color = NORMAL_COLOR
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert to RGB for emotion detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update visual indicator
            current_time = time.time()
            if is_detecting and current_time - detection_start_time > BLINK_DURATION:
                # Turn off detection indicator after BLINK_DURATION seconds
                is_detecting = False
                frame_border_color = NORMAL_COLOR
            
            # Check if it's time to detect emotions
            if current_time - last_detection_time > DETECTION_INTERVAL:
                # Start detection visual indicator
                is_detecting = True
                detection_start_time = current_time
                frame_border_color = DETECTION_COLOR
                print("Detecting emotions...")
                
                last_detection_time = current_time
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Update detected faces
                detected_faces = faces
                
                # Reset emotions list
                emotions = []
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = rgb_frame[y:y+h, x:x+w]
                    
                    # Resize for DeepFace
                    face_roi = cv2.resize(face_roi, (240, 240))
                    
                    try:
                        # Detect emotion
                        result = DeepFace.analyze(
                            face_roi, 
                            actions=['emotion'], 
                            detector_backend='ssd',
                            enforce_detection=False
                        )
                        
                        # Get dominant emotion
                        emotion = result[0]['dominant_emotion']
                        emotions.append(emotion)
                        
                        # Clean up
                        del face_roi
                        gc.collect()
                    except Exception as e:
                        print(f"Error analyzing face: {e}")
                
                # If faces were detected, blend emotions
                if len(faces) > 0:
                    # Calculate weights for each face
                    weights = calculate_face_weights(detected_faces, frame_width, frame_height)
                    
                    # Blend emotions based on weights
                    blended_emotion = blend_emotions(emotions, weights)
                    
                    # Update visualization
                    if seed_input and blended_emotion != current_blended_emotion:
                        current_blended_emotion = blended_emotion
                        new_seed_value = f"{blended_emotion}_{int(time.time())}"
                        seed_input.send_keys(Keys.CONTROL + "a")
                        time.sleep(0.1)
                        seed_input.send_keys(new_seed_value)
                        seed_input.send_keys(Keys.ENTER)
                    
                    # Play appropriate music
                    player.play_emotion(blended_emotion)
            
            # Draw border around frame to indicate detection
            border_thickness = 10 if is_detecting else 2
            cv2.rectangle(frame, (0, 0), (frame_width-1, frame_height-1), 
                        frame_border_color, border_thickness)
            
            # Draw faces and emotions on frame
            for i, (x, y, w, h) in enumerate(detected_faces):
                # Draw rectangle around face - use detection color when actively detecting
                face_color = frame_border_color if is_detecting else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
                
                # Display emotion if available
                if i < len(emotions):
                    cv2.putText(
                        frame, emotions[i], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2
                    )
            
            # Display detection status
            if is_detecting:
                cv2.putText(
                    frame, "DETECTING EMOTIONS", (frame_width//2 - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, DETECTION_COLOR, 2
                )
            
            # Display blended emotion with larger text
            cv2.putText(
                frame, f"Blended: {current_blended_emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2
            )
            
            # Display the frame in the resized window
            cv2.imshow(WINDOW_NAME, frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        pygame.midi.quit()
        driver.quit()

if __name__ == "__main__":
    main()