# Stroke_Detection-and-Segmentation-by-Using-Machine-Learning

*In this project, a convolutional neural network (CNN) based amd Machine Learningarchitecture was developed to classify stroke CT images into three categories: non-stroke, hemorrhage, and ischemic. In addition, classical machine learning techniques such as U-Net were used for image segmentation tasks. The project includes preprocessing, training, and evaluation stages with accuracy, weighted F1-score, confusion matrix, and ROC curve analyses. 
# Stroke Detection and Segmentation by Using Machine Learning

This project focuses on detecting and segmenting stroke regions from CT images by applying classical image processing techniques, feature extraction methods,Machine Learning  and Deep Learning models.

## Project Structure

- [`Image_Processing_Techniques`](https://github.com/Ibrahimumutdoruk/Stroke_Detection-and-Segmntation-by-Using-Machine-Learning/tree/main/Image_Processing_Techniques):  
  Classical image processing methods such as thresholding, morphological operations, CLAHE, and contour detection were applied for stroke segmentation. These techniques were used mainly for hemorrhage detection where clear contrast differences were present.

- [`Features`](https://github.com/Ibrahimumutdoruk/Stroke_Detection-and-Segmntation-by-Using-Machine-Learning/tree/main/Features):  
  Feature extraction techniques were implemented, including texture-based, intensity-based, and statistical feature calculations from segmented regions. Extracted features can be used to train traditional machine learning models like SVM, Random Forest, or XGBoost.

## 🧠 Models Used

### Convolutional Neural Networks (CNN)
- **ResNet18**
- **EfficientNetB0**

> Transfer learning was applied using pretrained ImageNet weights. Input images were resized to 512x512 and normalized. Models were trained for 10 epochs using Adam optimizer and evaluated using accuracy, weighted F1-score, confusion matrix, and ROC curve analysis.

### Classical Machine Learning Algorithms
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**

These algorithms were trained on features extracted from segmented stroke regions using statistical, texture, and intensity metrics.

---

## Dataset

The CT stroke dataset used in this project can be accessed from the [Türkiye Ministry of Health Open Data Portal](https://acikveri.saglik.gov.tr/Home/DataSets?categoryId=10).

## Requirements

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

## 🛠 Installation

```bash
git clone https://github.com/Ibrahimumutdoruk/Stroke_Detection-and-Segmentation-by-Using-CNN-ML.git
cd Stroke_Detection-and-Segmentation-by-Using-CNN-ML
pip install -r requirements.txt
> ⚠️ **Important**: Make sure to update the `dataset_path` variable in each script before execution.

```bash
cd Features
python feature_extraction_script.py
```

---

## 🧠 CNN Model Training

### ResNet18
```bash
cd ResNet18_Classification
python train_resnet.py
```

### EfficientNetB0
```bash
cd EfficientNet_Classification
python train_efficientnet.py
```

---

## ⚙️ Machine Learning Training

```bash
cd ML_Models
python train_logistic_regression.py
python train_svm.py
python train_random_forest.py
## Notes

- Classical methods achieved good results for hemorrhage segmentation, but ischemic stroke segmentation remains challenging due to low contrast differences.
- Assistance from radiologists may be required for accurate ischemic stroke segmentation.
- The U-Net segmentation model was also trained; however, due to insufficient data, it did not achieve high performance. It can be further developed for better results.
## 📊 Evaluation Metrics

- Accuracy  
- Weighted F1-Score  
- Confusion Matrix  
- ROC Curve and AUC Score
## ⚠️ Notes

- Classical image processing techniques perform well for hemorrhage segmentation due to high contrast.
- Ischemic stroke segmentation remains challenging due to low contrast; expert support may be needed.
- U-Net was tested for segmentation but underperformed due to limited annotated data.
- Don’t forget to **update dataset paths** (`dataset_path`) in each script before running.

---
---

> 🔍 This hybrid approach enables robust classification and segmentation of medical images for stroke diagnosis.
```
---

![Screenshot 2025-07-08 015633](https://github.com/user-attachments/assets/af8ff316-8495-40a0-83c5-a9347939d367)
![Screenshot 2025-07-08 015442](https://github.com/user-attachments/assets/b45eef24-f6ac-4d90-be6f-317cbf5235c5)
![Screenshot 2025-07-08 015525](https://github.com/user-attachments/assets/f74d9a3e-07dd-411d-b814-6813ddfb9989)

