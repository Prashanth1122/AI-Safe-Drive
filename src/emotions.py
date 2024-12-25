import numpy as np
import cv2
import pygame
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time

# Initialize pygame mixer for music and horn sound playback
pygame.mixer.init()

# Emotion to music mood mapping
emotion_to_music = {
    "Angry": "angry",
    "Disgusted": "neutral",
    "Fearful": "neutral",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "neutral"
}

# Function to play music based on detected emotion
def play_music(emotion, current_emotion, last_change_time, cooldown_time=5):
    # Check if enough time has passed since the last emotion change
    if time.time() - last_change_time >= cooldown_time:
        if emotion != current_emotion:  # If emotion changes, stop current music and play new one
            pygame.mixer.music.stop()  # Stop any currently playing music
            music_folder = emotion_to_music.get(emotion, "neutral")
            
            # Get all mp3 files in the folder
            music_files = [f for f in os.listdir(f'music/{music_folder}') if f.endswith('.mp3')]
            
            if music_files:
                # Pick a random song from the list
                song = random.choice(music_files)
                song_path = os.path.join('music', music_folder, song)
                print(f"Playing {song_path} for {emotion} mood")
                
                # Play the new music
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play(-1)  # Play on loop
            else:
                print(f"No songs found for {music_folder} mood")
            
            # Update the current emotion and reset the change timer
            current_emotion = emotion
            last_change_time = time.time()

    return current_emotion, last_change_time

# Emotion detection using pretrained model
def detect_emotion_and_sleep():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('model.h5')  # Load your pre-trained model

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    cap = cv2.VideoCapture(0)
    current_emotion = None
    last_change_time = time.time()  # Track the time of the last emotion change
    eyes_closed_counter = 0  # Counter to track eyes closed
    threshold = 10  # Number of frames for eyes to be closed to detect sleep

    # Load Haar Cascade for face and eye detection
    face_casc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyes_casc = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Check if Haar Cascade files are loaded correctly
    if face_casc.empty() or eyes_casc.empty():
        print("Error loading cascade files.")
        exit()

    # Function to check if eyes are closed (sleep detection)
    def is_sleeping(eyes_closed_counter, threshold=10):
        """Returns True if eyes are closed for a prolonged period (threshold reached)."""
        if eyes_closed_counter >= threshold:
            return True  # Person is likely sleeping
        return False

    # Function to play horn sound for sleep detection
    def play_horn_sound():
        horn_sound = 'horn.mp3'  # Ensure you have a horn.mp3 file in the same directory
        if os.path.exists(horn_sound):
            pygame.mixer.music.load(horn_sound)
            pygame.mixer.music.play(0)  # Play the horn sound once
            print("Sleep detected, playing horn sound!")
        else:
            print("Horn sound file not found!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_casc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            detected_emotion = emotion_dict[maxindex]
            cv2.putText(frame, detected_emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Play the music based on detected emotion
            current_emotion, last_change_time = play_music(detected_emotion, current_emotion, last_change_time)

            # Detect eyes for sleep detection
            eyes = eyes_casc.detectMultiScale(roi_gray)

            if len(eyes) == 0:  # If no eyes detected, likely eyes are closed
                eyes_closed_counter += 1
            else:
                eyes_closed_counter = 0  # Reset counter if eyes are detected

            # If sleep is detected (eyes closed for a prolonged period)
            if is_sleeping(eyes_closed_counter, threshold):
                play_horn_sound()  # Play the horn sound

        cv2.imshow('Emotion and Sleep Detection', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion_and_sleep()
