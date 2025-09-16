import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veriyi oku
df = pd.read_csv(r"D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\Features\all_features_balanced.csv")

print(df.columns)

# 2. Özellik ve etiket ayır
X = df.drop(columns=["filename", "label", "Disorder"])
y = df["label"]

clf = LogisticRegression(
    penalty='l2',          # Ridge regularizasyon
    C=1.0,                 # Düzenleme parametresi (1.0: varsayılan)
    solver='saga',         # Büyük veri setleri için iyi çalışır
    max_iter=1000,         # Daha yüksek iterasyon sınırı
    multi_class='multinomial',
    random_state=42
)

class_names = ["non-stroke", "ischemic", "hemorrhage"]

# 3. K-Fold ayarları
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)
train_accuracies, val_accuracies = [], []
train_f1s, val_f1s = [], []
all_y_true, all_y_pred = [], []

# 4. Fold'lar üzerinde döngü
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='saga',
        max_iter=1000,
        multi_class='multinomial',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Tahminler
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    # Metrikler (train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_accuracies.append(train_acc)
    train_f1s.append(train_f1)

    # Metrikler (validation)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)

    all_y_true.extend(y_val)
    all_y_pred.extend(y_val_pred)

    print(f"Fold {fold+1}")
    print(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
    print("-" * 40)

# 5. Ortalama sonuçlar
print("==== Ortalama Sonuçlar ====")
print(f"Train Accuracy (avg): {np.mean(train_accuracies):.4f}")
print(f"Train F1 Score  (avg): {np.mean(train_f1s):.4f}")
print(f"Val   Accuracy (avg): {np.mean(val_accuracies):.4f}")
print(f"Val   F1 Score  (avg): {np.mean(val_f1s):.4f}")

# 6. Confusion Matrix (sınıf isimleriyle)
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Tüm Fold'lar)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Classification raporu
print("\nClassification Report:\n", classification_report(all_y_true, all_y_pred, target_names=class_names))

# 8. Genel Accuracy ve F1
final_acc = accuracy_score(all_y_true, all_y_pred)
final_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
print(f"\n Final Accuracy on Full Validation Set: {final_acc:.4f}")
print(f" Final Weighted F1 Score: {final_f1:.4f}")

# 9. Feature Importance yerine Koef. büyüklüğü
coef = np.mean(np.abs(clf.coef_), axis=0)  # Ortalamalanmış mutlak katsayı
feat_names = X.columns
importance_df = pd.DataFrame({"Feature": feat_names, "Importance": coef})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# 10. Çizim
plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
plt.title(" Top 15 Coefficients (Importance)")
plt.tight_layout()
plt.show()

# 11. Terminal çıktısı (opsiyonel)
print("\n Most Important Features (Coefficient Magnitude):")
print(importance_df.head(10).to_string(index=False))
