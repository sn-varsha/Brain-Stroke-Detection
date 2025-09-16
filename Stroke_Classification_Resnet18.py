# ==============================
# Stroke Classification (ResNet18)
# ==============================

# Gerekli kütüphaneler
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
root_dir = r'D:\Stroke_Detection-and-Segmentation-by-Using-CNN-ML\all_png_images'  # Burayı kendi kullanıcı adına göre değiştir!
# image_dir = os.path.join(root_dir, 'images')
image_dir = root_dir

# ---------- Dataset Sınıfı ----------
class StrokeDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        class_to_idx = {'non-stroke': 0, 'hemorrhage': 1, 'ischemic': 2}

        for cls in os.listdir(image_root):
            cls_folder = os.path.join(image_root, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # RGB'ye çeviriyoruz
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ---------- Transformasyonlar ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- Dataset ve DataLoader ----------
dataset = StrokeDataset(image_root=image_dir, transform=transform)

# Dataseti %80 train %20 test şeklinde bölüyoruz
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------- Model: Pretrained ResNet18 ----------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 sınıf (non-stroke, hemorrhage, ischemic)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ---------- Loss ve Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------- Eğitim ----------
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ---------- Test + Grafiklerle Evaluation ----------
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Skorlar
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')

print("\n=== Test Sonuçları ===")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Weighted F1-Score: {f1*100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=['non-stroke', 'hemorrhage', 'ischemic'], digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['non-stroke', 'hemorrhage', 'ischemic'], yticklabels=['non-stroke', 'hemorrhage', 'ischemic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
all_labels_np = np.array(all_labels)
all_probs_np = np.array(all_probs)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((all_labels_np == i).astype(int), all_probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC Plot
plt.figure(figsize=(8,6))
colors = ['blue', 'green', 'red']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid()
plt.show()
