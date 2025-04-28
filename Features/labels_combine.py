import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy, skew, kurtosis


# Tüm sınıfların feature dosyalarını oku
df_iskemi = pd.read_csv("features_iskemi.csv")
df_kanama = pd.read_csv("features_kanama.csv")
df_yok = pd.read_csv("features_iskemi_yok.csv")

# Label ve hastalık kolonu ekle
df_iskemi["label"] = 1
df_iskemi["hastalik"] = "iskemi"

df_kanama["label"] = 2
df_kanama["hastalik"] = "kanama"

df_yok["label"] = 0
df_yok["hastalik"] = "yok"

# Tümünü birleştir
df_all = pd.concat([df_iskemi, df_kanama, df_yok], ignore_index=True)

# Weka için en son label olması iyidir
weka_ready = df_all[[*df_all.columns.difference(['label', 'hastalik', 'filename']), 'label', 'hastalik', 'filename']]
weka_ready.to_csv("features_all_cleaned.csv", index=False)

print("✅ Tüm sınıflar birleştirildi, normalize edildi ve Weka için uygun CSV dosyası hazırlandı.")