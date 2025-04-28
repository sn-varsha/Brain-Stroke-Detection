import cv2
import numpy as np
import os

OVERLAY_DIR = '/Users/emin/Library/CloudStorage/GoogleDrive-eminsoylemezzz@gmail.com/My Drive/1. PROJECTS/BioMedical Dataset/İNME VERİ SETİ/Eğitim Veri Seti_İskemi/İskemi Veri Seti/OVERLAY'
MASK_SAVE_DIR = '/Users/emin/Library/CloudStorage/GoogleDrive-eminsoylemezzz@gmail.com/My Drive/1. PROJECTS/BioMedical Dataset/İNME VERİ SETİ/Eğitim Veri Seti_İskemi/İskemi Veri Seti/MASK'  # yeni oluşturulacak klasör
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

for fname in os.listdir(OVERLAY_DIR):
    if not fname.endswith(".png"):
        continue

    path = os.path.join(OVERLAY_DIR, fname)
    img = cv2.imread(path)

    # BGR'den HSV'ye dönüştürme
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Kırmızı renk aralıkları (HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Maskeleri oluşturma
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Maskeyi kaydetme
    save_path = os.path.join(MASK_SAVE_DIR, fname)
    cv2.imwrite(save_path, mask)

    print(f"{fname} işlendi ve maskesi kaydedildi.")