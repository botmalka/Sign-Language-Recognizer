import cv2
import numpy as np

IMAGE_FILES = [r"C:\Users\Tori\Documents\ASL Detector\asl_dataset\a\hand1_a_bot_seg_1_cropped.jpeg", r"C:\Users\Tori\Documents\ASL Detector\asl_dataset\a\hand1_a_bot_seg_2_cropped.jpeg"]

dimensions = (800,800)



for idx, file in enumerate(IMAGE_FILES):
  # Read an image, flip it around y-axis for correct handedness output (see
  # above).
  print(file)
  img = cv2.imread(file)
  resized = cv2.resize(img, dimensions, interpolation = cv2.INTER_AREA)
  #image = cv2.flip(cv2.imread(file), 1)
  cv2.imshow("window", img)
  cv2.waitKey(5)


  
while True:

    if 0xFF == 27:
      break

cv2.destroyAllWindows()