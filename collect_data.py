import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# Letters to collect
TARGET_LETTERS = ['R', 'U', 'T', 'H']
SAMPLES_PER_LETTER = 200  # Number of frames to capture per letter
CSV_FILENAME = 'asl_data.csv'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_styles
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# Check if CSV exists; if not, initialize dataframe
if not os.path.exists(CSV_FILENAME):
    columns = [f'k_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
    pd.DataFrame(columns=columns).to_csv(CSV_FILENAME, index=False)

print("Starting Data Collection for letters:", TARGET_LETTERS)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    for letter in TARGET_LETTERS:
        print(f"\nPrepare to sign letter: '{letter}'")
        print("Press 's' when you are holding the sign to start recording frames.")
        
        recording = False
        sample_count = 0

        while cap.isOpened() and sample_count < SAMPLES_PER_LETTER:
            success, image = cap.read()
            if not success:
                continue

            # Flip & convert image
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            status_text = f"Letter: {letter} | Status: {'RECORDING' if recording else 'Press S to Start'}"
            count_text = f"Progress: {sample_count}/{SAMPLES_PER_LETTER}"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # If recording, extract landmarks
                    if recording:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks.append(letter)

                        # Save row to CSV
                        df = pd.DataFrame([landmarks])
                        df.to_csv(CSV_FILENAME, mode='a', header=False, index=False)
                        sample_count += 1

            # Draw HUD overlay
            cv2.putText(image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if recording else (0, 165, 255), 2)
            cv2.putText(image, count_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('ASL Data Collector', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                recording = True
            elif key == ord('q'):
                break

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"\nData collection complete! Data saved to {CSV_FILENAME}.")