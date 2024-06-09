
# Import the necessary libraries
import time
import os
import gc

import pygame
import cv2 
from deepface import DeepFace
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


# Next 2 lines are needed to specify the path to your geckodriver
geckodriver_path = "/snap/bin/geckodriver"
driver_service = webdriver.FirefoxService(executable_path=geckodriver_path)

driver = webdriver.Firefox(service=driver_service)
# browser = webdriver.Firefox() # Originial Example

#driver.get("http://172.16.25.179:9966/")
driver.get("http://127.0.0.1:9966/")

seed_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")

# Initialize pygame mixer
pygame.mixer.init()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# backend for the deepface
backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

# Function to play sound based on emotion with crossfade
def play_emotion_sound(emotion):
    sound_files = {
        'neutral': 'do.wav',
        'sad': 're.wav',
        'happy': 'mi.wav',
        'angry': 'fa.wav',
        'fear': 'sol.wav',
        'surprise': 'la.wav',
        'disgust': 'ti.wav'
    }
    if emotion in sound_files:
        sound_file = sound_files[emotion]
        if os.path.exists(sound_file):
            if pygame.mixer.music.get_busy():  # Check if music is playing
                pygame.mixer.music.fadeout(500)  # Fade out the current sound over 0.5 seconds
            pygame.mixer.music.load(sound_file)

            new_seed_value = emotion  + str("425861qt47z") # Replace with the seed value you want
            #time.sleep(0.1)
            #seed_input.clear()
            seed_input.send_keys(Keys.CONTROL + "a")
            time.sleep(0.1)

            # Enter the new seed value
            
            seed_input.send_keys(new_seed_value) 
            #time.sleep(0.1)
            seed_input.send_keys(Keys.ENTER) # You might use Keys.RETURN or other keys as appropriate

            pygame.mixer.music.play(fade_ms=500)            
            
        else:
            print(f"Sound file {sound_file} not found.")

# Initialize variables to keep track of detected faces and emotions
detected_faces = []
emotions = []

# Initialize a variable to keep track of time for interval control
last_detection_time = 0
detection_interval = 6  # seconds

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Get the current time
    current_time = time.time()

    # Check if it's time to run the emotion analysis
    if current_time - last_detection_time > detection_interval:
        last_detection_time = current_time

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        detected_faces = faces  # Update the detected faces

        emotions = []  # Reset emotions list
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]
            #resize the frame
            # Define the desired width and height
            width = 240
            height = 240
            # Resize the frame
            face_roi = cv2.resize(face_roi, (width, height))
            
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], detector_backend = backends[0], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']
            emotions.append(emotion)  # Store the detected emotion

            # Play sound based on emotion
            play_emotion_sound(emotion)

            # Release memory
            del face_roi
            gc.collect()

    # Draw rectangles and labels for all detected faces
    for i, (x, y, w, h) in enumerate(detected_faces):
        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if i < len(emotions):
            cv2.putText(frame, emotions[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
gc.collect()

