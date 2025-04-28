import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy, skew, kurtosis

IMG_DIR = "/Users/emin/Library/CloudStorage/GoogleDrive-eminsoylemezzz@gmail.com/My Drive/1. PROJECTS/BioMedical Dataset/İNME VERİ SETİ/Eğitim Veri Seti İnme Yok_kronik süreç_PNG/İnme Yok_kronik süreç_diğer Veri Set_PNG"
features_list = []

for fname in sorted(os.listdir(IMG_DIR)):
    img_path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # Normalize et (0-255 aralığına)
    region = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    region = region.astype(np.uint8)
    
    if region.size == 0:
        continue

    # İstatistiksel özellikler
    f_mean = np.mean(region)
    f_std = np.std(region)
    f_var = np.var(region)
    f_median = np.median(region)
    f_iqr = np.percentile(region, 75) - np.percentile(region, 25)
    f_mad = np.median(np.abs(region - np.median(region)))
    f_skew = skew(region.flatten())
    f_kurt = kurtosis(region.flatten())
    f_entropy = entropy(np.histogram(region, bins=256)[0] + 1)
    f_min = np.min(region)
    f_max = np.max(region)
    f_range = f_max - f_min

    # Histogram bazlı
    hist = np.histogram(region, bins=16)[0]
    f_hist_mean = np.mean(hist)
    f_hist_std = np.std(hist)
    f_hist_entropy = entropy(hist + 1)
    f_hist_skew = skew(hist)
    f_hist_kurt = kurtosis(hist)
    f_hist_max_bin = np.max(hist)

    # GLCM doku özellikleri
    distances = [2, 3, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(region, distances=distances, angles=angles, symmetric=True, normed=True)
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

# Veriyi dataframe'e çevir
df_iskemi_yok = pd.DataFrame(features_list, columns=columns)
df_iskemi_yok.to_csv("features_iskemi_yok.csv", index=False)
print("✅ Dummy feature'lar çıkarıldı ve normalize edilmiş 'İskemi Yok' feature'ları kaydedildi.")