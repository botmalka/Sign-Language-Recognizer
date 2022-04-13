import cv2
import mediapipe as mp
from math import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#imports normalized tabular dataset as dataframe from DataReader.py and adds column names
asl_dataset_directory = r"C:\Users\Tori\Documents\ASL Detector\asl_dataset"  
name_list = ["Column_" + str(i) for i in range(0, 21)]
df = pd.read_csv(asl_dataset_directory + r"\..\asl_dataset.csv", header=None, names=name_list)
df = df.copy()
target = df.columns[[0]]
y = df[target]
features = df.columns.drop('Column_0')
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) #random_state=42

#defines model
model = KNeighborsClassifier(n_neighbors=26)
model.fit(X_train, y_train)
model.predict(X_test)

#defines mediapipe functions 
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
            #ignores empty video frames or glitches
            continue

        #to improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
    
        #draw the hand annotations on the image
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
                
                angle_list = [
                    angle(wrist, thumb_tip), #datapoints 1-20, as angles between points
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

                current_hand_shape = model.predict([angle_list])
                print(current_hand_shape[0])
            
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                #flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

#destroys the opencv windows and turns off the webcam
cap.release()
cv2.destroyAllWindows()