import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy, skew, kurtosis


# Tüm sınıfların feature dosyalarını oku
df_ischemic = pd.read_csv("features_ischemic.csv")
df_hemorrhage = pd.read_csv("features_hemorrhage.csv")
df_nonstroke = pd.read_csv("features_ischemic_non-stroke.csv")

# Label ve hastalık kolonu ekle
df_ischemic["label"] = 1
df_ischemic["Disorder"] = "ischemic"

df_hemorrhage["label"] = 2
df_hemorrhage["Disorder"] = "hemorrhage"

df_nonstroke["label"] = 0
df_nonstroke["Disorder"] = "ynon-stroke"

# Tümünü birleştir
df_all = pd.concat([df_ischemic, df_hemorrhage, df_nonstroke], ignore_index=True)

# Weka için en son label olması iyidir
weka_ready = df_all[[*df_all.columns.difference(['label', 'Disorder', 'filename']), 'label', 'Disorder', 'filename']]
weka_ready.to_csv("features_all_cleaned.csv", index=False)

print("All classes were combined, normalized, and a CSV file suitable for Weka was prepared.")