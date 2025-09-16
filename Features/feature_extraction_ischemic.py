import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy, skew, kurtosis

IMG_DIR = r"D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\ischemic\PNG"  # Beyin görüntüleri
MASK_DIR = r"D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\ischemic\MASK"   # Maskelerin bulunduğu klasör
features_list = []

for fname in sorted(os.listdir(IMG_DIR)):
    img_path = os.path.join(IMG_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname)  # Maske adı ile görüntü adı aynı varsayılıyor

    # Görüntü ve maskeyi oku
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    # Maskeyi uygula
    region = cv2.bitwise_and(img, img, mask=mask)

    # Maskelenmiş bölgedeki sıfır olmayan pikselleri al
    masked_pixels = region[mask > 0]

    if masked_pixels.size == 0:
        continue

    # İstatistiksel özellikler
    f_mean = np.mean(masked_pixels)
    f_std = np.std(masked_pixels)
    f_var = np.var(masked_pixels)
    f_median = np.median(masked_pixels)
    f_iqr = np.percentile(masked_pixels, 75) - np.percentile(masked_pixels, 25)
    f_mad = np.median(np.abs(masked_pixels - np.median(masked_pixels)))
    f_skew = skew(masked_pixels.flatten())
    f_kurt = kurtosis(masked_pixels.flatten())
    f_entropy = entropy(np.histogram(masked_pixels, bins=256)[0] + 1)
    f_min = np.min(masked_pixels)
    f_max = np.max(masked_pixels)
    f_range = f_max - f_min

    # Histogram bazlı
    hist = np.histogram(masked_pixels, bins=16)[0]
    f_hist_mean = np.mean(hist)
    f_hist_std = np.std(hist)
    f_hist_entropy = entropy(hist + 1)
    f_hist_skew = skew(hist)
    f_hist_kurt = kurtosis(hist)
    f_hist_max_bin = np.max(hist)

    # GLCM doku özellikleri (region sadece maske içindeki değerleri barındırmalı)
    region_masked = np.zeros_like(region)
    region_masked[mask > 0] = region[mask > 0]

    distances = [2, 3, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # GLCM için boyut küçültme ve normalizasyon (256 seviyeden daha az seviyeye)
    region_quantized = cv2.normalize(region_masked, None, alpha=0, beta=15, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    glcm = graycomatrix(region_quantized, distances=distances, angles=angles, symmetric=True, normed=True)
    f_contrast_angles = [np.mean(graycoprops(glcm, 'contrast')[:, i]) for i in range(4)]
    f_homogeneity_angles = [np.mean(graycoprops(glcm, 'homogeneity')[:, i]) for i in range(4)]
    f_correlation_angles = [np.mean(graycoprops(glcm, 'correlation')[:, i]) for i in range(4)]
    f_asm_angles = [np.mean(graycoprops(glcm, 'ASM')[:, i]) for i in range(4)]
    f_energy_angles = [np.mean(graycoprops(glcm, 'energy')[:, i]) for i in range(4)]

    features_list.append([
        fname, f_mean, f_std, f_var, f_median, f_iqr, f_mad, f_skew, f_kurt, f_entropy,
        f_min, f_max, f_range, f_hist_mean, f_hist_std, f_hist_entropy,
        f_hist_skew, f_hist_kurt, f_hist_max_bin,
        *f_contrast_angles, *f_homogeneity_angles, *f_correlation_angles,
        *f_asm_angles, *f_energy_angles
    ])

columns = ['filename', 'mean', 'std', 'var', 'median', 'iqr', 'mad', 'skew', 'kurtosis', 'entropy',
           'min', 'max', 'range', 'hist_mean', 'hist_std', 'hist_entropy',
           'hist_skew', 'hist_kurtosis', 'hist_max_bin',
           'contrast_0', 'contrast_45', 'contrast_90', 'contrast_135',
           'homogeneity_0', 'homogeneity_45', 'homogeneity_90', 'homogeneity_135',
           'correlation_0', 'correlation_45', 'correlation_90', 'correlation_135',
           'ASM_0', 'ASM_45', 'ASM_90', 'ASM_135',
           'energy_0', 'energy_45', 'energy_90', 'energy_135']

# Veriyi dataframe'e çevir ve kaydet
df_features = pd.DataFrame(features_list, columns=columns)
df_features.to_csv("features_ischemic.csv", index=False)
print("Features were extracted from the masked region and saved.")