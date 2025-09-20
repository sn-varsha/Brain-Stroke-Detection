import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Read the data
df = pd.read_csv(r"D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\Features\all_features_balanced.csv")

# 2. Separate features and labels
X = df.drop(columns=["filename", "label", "Disorder"])
y = df["label"]

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,              # Trees are shallower
    min_samples_split=20,     # Requires more data before splitting a branch
    min_samples_leaf=10,      # There must be at least 10 samples per leaf
    max_features='sqrt',      # Use fewer features in each split
    bootstrap=True,           # Train each tree with a different subset
    random_state=42
)
# Match class names
class_names = ["non-stroke", "ischemic", "hemorrhage"]

# 3. K-Fold settings
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)
train_accuracies, val_accuracies = [], []
train_f1s, val_f1s = [], []
all_y_true, all_y_pred = [], []

# 4. Loop over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    # Metrics (train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_accuracies.append(train_acc)
    train_f1s.append(train_f1)

    # Metrics (validation)
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

# 5. Average results
print("==== Ortalama Sonuçlar ====")
print(f"Train Accuracy (avg): {np.mean(train_accuracies):.4f}")
print(f"Train F1 Score  (avg): {np.mean(train_f1s):.4f}")
print(f"Val   Accuracy (avg): {np.mean(val_accuracies):.4f}")
print(f"Val   F1 Score  (avg): {np.mean(val_f1s):.4f}")

# 6. Confusion Matrix (with class names)
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (All Folds)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Classification report
print("\n Classification Report:\n", classification_report(all_y_true, all_y_pred, target_names=class_names))

# 8. Overall Accuracy and F1
final_acc = accuracy_score(all_y_true, all_y_pred)
final_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
print(f"\n Final Accuracy on Full Validation Set: {final_acc:.4f}")
print(f" Final Weighted F1 Score: {final_f1:.4f}")

# 9. Feature Importance
importances = clf.feature_importances_
feat_names = X.columns
importance_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# 10. Plot
plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
plt.title(" Top 15 Feature Importances")
plt.tight_layout()
plt.show()

# 11. Print to terminal if desired (optional)
print("\n Most Important Features:")
print(importance_df.head(10).to_string(index=False))