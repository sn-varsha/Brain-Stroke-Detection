
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from scipy.stats import entropy, skew, kurtosis
import mahotas
from tqdm import tqdm
from scipy.fft import fft2
import pywt

def extract_massive_features(img):
    features = {}
    img = cv2.resize(img, (256, 256))

    # Histogram özellikleri
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (np.sum(hist) + 1e-6)
    features.update({
        "hist_entropy": entropy(hist_norm),
        "hist_skew": skew(hist_norm),
        "hist_kurtosis": kurtosis(hist_norm),
        "hist_mean": np.mean(hist),
        "hist_std": np.std(hist),
    })

    # İstatistiksel
    features.update({
        "mean": np.mean(img),
        "std": np.std(img),
        "min": np.min(img),
        "max": np.max(img),
        "rms": np.sqrt(np.mean(img**2)),
        "var": np.var(img)
    })

    # GLCM (çok açı, mesafe)
    distances = [1, 2, 3, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for d in distances:
        for a in angles:
            glcm = graycomatrix(img, distances=[d], angles=[a], symmetric=True, normed=True)
            for prop in props:
                features[f'glcm_{prop}_d{d}_a{int(np.rad2deg(a))}'] = graycoprops(glcm, prop)[0, 0]

    # LBP
    lbp = local_binary_pattern(img, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    for i, val in enumerate(lbp_hist):
        features[f'lbp_{i}'] = val

    # Haralick (mahotas)
    haralick_feats = mahotas.features.haralick(img).mean(axis=0)
    for i, val in enumerate(haralick_feats):
        features[f'haralick_{i}'] = val

    # HOG
    hog_features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    for i in range(min(256, len(hog_features))):  # truncate to 256
        features[f'hog_{i}'] = hog_features[i]

    # Gabor
    ksize = 31
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for freq in (0.1, 0.2, 0.3, 0.4):
            kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, freq*ksize, 0.5, 0, ktype=cv2.CV_32F)
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            features[f'gabor_mean_t{int(theta*100)}_f{int(freq*10)}'] = np.mean(fimg)
            features[f'gabor_std_t{int(theta*100)}_f{int(freq*10)}'] = np.std(fimg)

    # Fourier
    f_transform = np.abs(fft2(img))
    features.update({
        "fft_mean": np.mean(f_transform),
        "fft_std": np.std(f_transform),
        "fft_max": np.max(f_transform),
    })

    # Wavelet
    coeffs2 = pywt.dwt2(img, 'db1')
    LL, (LH, HL, HH) = coeffs2
    for mat, name in zip([LL, LH, HL, HH], ["LL", "LH", "HL", "HH"]):
        features[f'wavelet_{name}_mean'] = np.mean(mat)
        features[f'wavelet_{name}_std'] = np.std(mat)

    return features

# Ana dizin
base_path = "/Users/emin/Desktop/Biomedical/dataset"
classes = ["iskemi", "kanama", "yok"]

data = []

for label, cls in enumerate(classes):
    class_path = os.path.join(base_path, cls)
    for fname in tqdm(os.listdir(class_path), desc=f"Processing {cls}"):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feats = extract_massive_features(img)
            feats["label"] = label
            feats["hastalik"] = cls
            feats["filename"] = fname
            data.append(feats)



df = pd.DataFrame(data)
df.to_csv("all_features.csv", index=False)
print("✅ Tüm özellikler çıkarıldı ve all_features.csv dosyasına kaydedildi.")
