import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Dizin Ayarları ----------
root_dir = r'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images'
# image_dir = os.path.join(root_dir, 'images')
image_dir = root_dir

# ---------- Dataset Sınıfı ----------
class StrokeDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        class_to_idx = {'non-stroke': 0, 'hemorrhage': 1, 'ischemic': 2}

        # for cls in os.listdir(image_root):
        #     cls_folder = os.path.join(image_root, cls)
        #     if os.path.isdir(cls_folder):
        #         for img_name in os.listdir(cls_folder):
        #             self.image_paths.append(os.path.join(cls_folder, img_name))
        #             self.labels.append(class_to_idx[cls])

        for cls in os.listdir(image_root):
            cls_folder = os.path.join(image_root, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    # Only add if it's a file and has a valid image extension
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------- Transformasyonlar ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Dataset ve Split ----------
dataset = StrokeDataset(image_dir, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------- Model ----------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ---------- Loss ve Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------- Eğitim ----------
num_epochs = 15  # Epoch sayısını 15 olarak belirledik
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    # Validation Accuracy
    val_accuracy = 100 * val_correct / val_total
    train_accuracy = 100 * train_correct / train_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Train Acc: {train_accuracy:.2f}% "
          f"Val Acc: {val_accuracy:.2f}%")

# ---------- Test ve Değerlendirme ----------
model.eval()
all_preds, all_labels, all_probs = [], [], []
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = 100 * test_correct / test_total
test_loss /= len(test_loader)

f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"\nTest Accuracy: {test_accuracy:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Weighted F1 Score: {f1*100:.2f}%\n")

print("Unique labels in test set:", np.unique(all_labels))
print("Unique predictions:", np.unique(all_preds))

print("Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=['non-stroke', 'hemorrhage', 'ischemic'], digits=4))
print(classification_report(all_labels, all_preds, labels=[0,1,2], target_names=['non-stroke', 'hemorrhage', 'ischemic'], digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['non-stroke', 'hemorrhage', 'ischemic'], yticklabels=['non-stroke', 'hemorrhage', 'ischemic'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC Curve
n_classes = 3
fpr, tpr, roc_auc = {}, {}, {}
labels_np = np.array(all_labels)
probs_np = np.array(all_probs)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((labels_np == i).astype(int), probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8,6))
colors = ['blue', 'green', 'red']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, color=colors[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# ---------- Model Kaydı ----------
model_path = os.path.join(root_dir, 'EfficiencyNet_stroke_classification_6600.pth')
torch.save(model.state_dict(), model_path)
print(f"The model has been saved: {model_path}")
