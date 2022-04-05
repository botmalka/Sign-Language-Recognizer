import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from math import sqrt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# # For static images:
# IMAGE_FILES = [r"C:\Users\Tori\Documents\HANDy\asl_dataset\a\hand1_a_bot_seg_1_cropped.jpeg"]
# with mp_hands.Hands(
#     static_image_mode=True, #set to True in original code
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

def scale(Ax, Ay, Bx, By, Cx, Cy):
    #this finds the distance between point B and the midpoint of line segment AC
    return round(sqrt(((float(Bx) - (float(Ax) + float(Cx))/2)**2) + (float(By) - (float(Ay) + float(Cy))/2)**2))

def distance(Ax, Ay, Bx, By):
    dist = sqrt(((Ay-By)**2)+((Ax-By)**2))
    return dist


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
        #print('hand_landmarks:', hand_landmarks)
        pinky = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
        wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
        pointer = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
        print(scale(pointer[0], pointer[1], wrist[0], wrist[1], pinky[0], pinky[1]))
        #print('pinky: ' + str(pinky[0]) + ', ' + str(pinky[1]))
        # print(f'wrist coordinates: (',
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width}, '
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height}, '
        #       f'{hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z} )'
        # )
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