import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

asl_dataset_directory = r"C:\Users\Tori\Documents\ASL Detector\asl_dataset"  
name_list = [i for i in range(0, 21)]
df = pd.read_csv(asl_dataset_directory + r"\..\asl_dataset.csv", header=None, names=name_list)
print(df.head())
df = df.copy()
target = df.columns[[0]]
features = df.drop(df.columns[[0]], axis=1, inplace=True)
print(df.head())
model = KNeighborsClassifier(n_neighbors=36)

#added for test purposes
lowest_distance, highest_distance = 999999, 0
lowest_scale, highest_scale = 999999, 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def scale(Ax, Ay, Bx, By, Cx, Cy):
#this finds the distance between point B and the midpoint of line segment AC
    return round(sqrt(((float(Bx) - (float(Ax) + float(Cx))/2)**2) + (float(By) - (float(Ay) + float(Cy))/2)**2),2)

def distance(point_1, point_2):
#standard Euclidian distance formula for points (point_1[0], point_1[1]) and (point_2[0], point_2[1])
    return round(sqrt((point_1[0]-point_2[0])**2 + (point_1[1] + point_2[1])**2), 2)

def angle(point_1, point_2):
#returns the slope between 2 points as a +/- decimal
    return round((point_1[1]-point_2[1])/(point_1[0]-point_2[0]), 4)

# For webcam input:
cap = cv2.VideoCapture(0)
image_height, image_width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
    
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                #assigns variables to point data as recognized by mediapipe for hand
                wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                thumb_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
                pointer_palm = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                pointer_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                middle_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
                ring_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)
                pinky_palm = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                pinky_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)
                #size_factor = scale(pinky_palm[0],pinky_palm[1],wrist[0],wrist[1],pointer_palm[0],pointer_palm[1]) * 0.3
                #size_factor = distance(pointer_palm, wrist) / 10
                #size_factor = 1
                
                sub = 'a'
                #creates a 20 item list to append to the dataframe, containing various distance points scaled to each hand
                angle_list = [
                    sub, #letter or number represented, taken from the folder name
                    angle(wrist, thumb_tip), #datapoints 1-20, scaled by hand size
                    angle(wrist, pointer_tip),
                    angle(wrist, middle_tip),
                    angle(wrist, ring_tip),
                    angle(wrist, pinky_tip),
                    angle(thumb_tip, pointer_tip),
                    angle(thumb_tip, middle_tip),
                    angle(thumb_tip, ring_tip),
                    angle(thumb_tip, pinky_tip),
                    angle(middle_tip, ring_tip),
                    angle(middle_tip, pinky_tip),
                    angle(pointer_palm, pointer_tip),
                    angle(pointer_palm, middle_tip),
                    angle(pointer_palm, ring_tip),
                    angle(pointer_palm, pinky_tip),
                    angle(pinky_palm, pointer_tip),
                    angle(pinky_palm, middle_tip),
                    angle(pinky_palm, ring_tip),
                    angle(pinky_palm, pinky_tip),
                    angle(pointer_palm, pinky_palm)
                ]
                #added for troubleshooting
                # if angle_list[-1] < lowest_distance:
                #     lowest_distance = angle_list[-1]
                # if angle_list[-1] > highest_distance:
                #     highest_distance = angle_list[-1]
                # if size_factor < lowest_scale:
                #     lowest_scale = size_factor
                # if size_factor > highest_scale:
                #     highest_scale = size_factor                  
                # print("lowest: " + str(lowest_distance))
                # print("highest: " + str(highest_distance))
                # print("ratio: " + str(lowest_distance/highest_distance))
                # print("low scale : " + str(lowest_scale))
                # print("high scale: "+ str(highest_scale))
                model.predict(angle_list)
                print(angle_list)
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()