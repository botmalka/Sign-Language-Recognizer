# Sign-Language-Recognizer
Recognized ASL letters from live camera using OpenCV, Mediapipe, and Tensorflow

This is a transfer-learning project built on top of Google's Mediapipe library. Mediapipe sets 

DataReader.py reads and converts any sign language image dataset to tabular data and outputs a CSV file (asl_dataset.csv). Mediapipe is used to collect points from the hand images and the slopes between 20 points are calculated

CameraReader.py uses OpenCV to read input from the camera, pandas for data manipulation, and skikit-learn to compare the current input to the tabular data collected by DataReader. A K-nearest neighbor algorithm is run to compare the visible hand to the dataset

![This is a sample of the application showing the letter H](https://raw.githubusercontent.com/botmalka/Sign-Language-Recognizer/main/ASL-H.png "The ASL letter H")
