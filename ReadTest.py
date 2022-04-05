import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from math import sqrt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def read_files():
    file_list, sub_list = [], []
    start_dir = r"C:\Users\Tori\Documents\ASL Detector\asl_dataset"                                                                                               
    subdirs = [x[0] for x in os.walk(start_dir)]                                                                      
    for subdir in subdirs:       
        print(subdir)                                                                                     
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:               
                #sub_list.append(str(subdir)[-1])                                                                       
                file_list.append(os.path.join(subdir, file)) 
                sub_list.append(str(subdir)[-1]) 
    zipped = zip(file_list, sub_list)
    return zipped
    
def scale(Ax, Ay, Bx, By, Cx, Cy):
#this finds the distance between point B and the midpoint of line segment AC
    return round(sqrt(((float(Bx) - (float(Ax) + float(Cx))/2)**2) + (float(By) - (float(Ay) + float(Cy))/2)**2),2)

def distance(point_1, point_2):
#standard Euclidian distance formula for points (point_1[0], point_1[1]) and (point_2[0], point_2[1])
    return round(sqrt((point_1[0]-point_2[0])**2 + (point_1[1] + point_2[1])**2), 2)

# For static images:
files_subs = read_files()
total_num_hands = 0
with mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
    for file, sub in files_subs:
        # Read an image, flip it around y-axis for correct handedness output (see above).

        total_num_hands = total_num_hands + 1
        img_flipped = cv2.imread(file)
        img = cv2.flip(img_flipped, 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        #skips to the next picture if no hand found
        if not results.multi_hand_landmarks:
            print("no hand found...")
            continue
        print(file + " from: " + sub)       
        image_height, image_width, _ = img.shape
        annotated_image = img.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
            thumb_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
            pointer_palm = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
            pointer_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
            middle_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
            ring_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)
            pinky_palm = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
            pinky_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)
            #print(scale(pinky_palm[0],pinky_palm[1],wrist[0],wrist[1],pointer_palm[0],pointer_palm[1]))           
            print(distance(pinky_tip, wrist))
        #     mp_drawing.draw_landmarks(
        #         annotated_image,
        #         hand_landmarks,
        #         mp_hands.HAND_CONNECTIONS,
        #         mp_drawing_styles.get_default_hand_landmarks_style(),
        #         mp_drawing_styles.get_default_hand_connections_style())
        # cv2.imwrite(r"C:\Users\Tori\Documents\ASL Detector\test.jpg", annotated_image)
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
print("number of hands: " +  str(total_num_hands))