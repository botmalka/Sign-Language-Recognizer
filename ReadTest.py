import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def read_files():
    file_list = []     
    start_dir = r"C:\Users\Tori\Documents\ASL Detector\asl_dataset"                                                                                               
    subdirs = [x[0] for x in os.walk(start_dir)]                                                                      
    for subdir in subdirs:       
        print(subdir)                                                                                     
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:                                                                                        
                file_list.append(os.path.join(subdir, file))                                                                         
    return file_list
    


# For static images:
image_files = read_files()
total_num_hands = 0
with mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.4) as hands:
    for i, file in enumerate(image_files):
        # Read an image, flip it around y-axis for correct handedness output (see above).
        print(file)
        total_num_hands = total_num_hands + 1
        img_flipped = cv2.imread(file)
        img = cv2.flip(img_flipped, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            print("no hand found...")
            continue
        image_height, image_width, _ = img.shape
        annotated_image = img.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            pinky = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
            wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
            pointer = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
        # print(scale(pointer[0], pointer[1], wrist[0], wrist[1], pinky[0], pinky[1]))
            
        #     print('hand_landmarks:', hand_landmarks)
        #     print(
        #         f'Index finger tip coordinates: (',
        #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        #         )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(r"C:\Users\Tori\Documents\ASL Detector\test.jpg", annotated_image)
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
print("number of hands: " +  str(total_num_hands))