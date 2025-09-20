import cv2
import numpy as np
import os

OVERLAY_DIR = 'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\ischemic\OVERLAY'
MASK_SAVE_DIR = 'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\ischemic\MASK'  # new folder to be created
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

for fname in os.listdir(OVERLAY_DIR):
    if not fname.endswith(".png"):
        continue

    path = os.path.join(OVERLAY_DIR, fname)
    img = cv2.imread(path)

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color ranges (HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Do not create the masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Do not save the mask
    save_path = os.path.join(MASK_SAVE_DIR, fname)
    cv2.imwrite(save_path, mask)

    print(f"{fname} processed and its mask was saved.")