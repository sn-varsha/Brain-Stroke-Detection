# Stroke_Detection-and-Segmentation-by-Using-Machine-Learning

*In this project, a convolutional neural network (CNN) based on ResNet18 architecture was developed to classify stroke CT images into three categories: non-stroke, hemorrhage, and ischemic. In addition, classical machine learning techniques such as U-Net were used for image segmentation tasks. The project includes preprocessing, training, and evaluation stages with accuracy, weighted F1-score, confusion matrix, and ROC curve analyses. 
# Stroke Detection and Segmentation by Using Machine Learning

This project focuses on detecting and segmenting stroke regions from CT images by applying classical image processing techniques, feature extraction methods, and deep learning models.

## Project Structure

- [`Image_Processing_Techniques`](https://github.com/Ibrahimumutdoruk/Stroke_Detection-and-Segmntation-by-Using-Machine-Learning/tree/main/Image_Processing_Techniques):  
  Classical image processing methods such as thresholding, morphological operations, CLAHE, and contour detection were applied for stroke segmentation. These techniques were used mainly for hemorrhage detection where clear contrast differences were present.

- [`Features`](https://github.com/Ibrahimumutdoruk/Stroke_Detection-and-Segmntation-by-Using-Machine-Learning/tree/main/Features):  
  Feature extraction techniques were implemented, including texture-based, intensity-based, and statistical feature calculations from segmented regions. Extracted features can be used to train traditional machine learning models like SVM, Random Forest, or XGBoost.

- `ResNet18_Classification`:  
  A Convolutional Neural Network (CNN) based on ResNet18 was used to classify CT images into three categories: non-stroke, hemorrhage, and ischemic stroke. The dataset consisted of 3000 images (1000 per class). 80% of the dataset was used for training, and 20% for testing. Transfer learning was applied using pretrained ImageNet weights. The model was evaluated with accuracy, weighted F1-score, confusion matrix, and ROC curve analyses.

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

1. Clone the repository:
```bash
git clone https://github.com/Ibrahimumutdoruk/Stroke_Detection-and-Segmntation-by-Using-Machine-Learning.git
```

2. Navigate to the project directory:
```bash
cd Stroke_Detection-and-Segmntation-by-Using-Machine-Learning
```

3. To apply classical image processing methods:
```bash
cd Image_Processing_Techniques
python processing_script.py
```

4. To perform feature extraction:
```bash
cd Features
python feature_extraction_script.py
```

5. To train and evaluate the ResNet18 model:
```bash
cd ResNet18_Classification
python train_resnet.py
```

## Notes

- Classical methods achieved good results for hemorrhage segmentation, but ischemic stroke segmentation remains challenging due to low contrast differences.
- Assistance from radiologists may be required for accurate ischemic stroke segmentation.
- The U-Net segmentation model was also trained; however, due to insufficient data, it did not achieve high performance. It can be further developed for better results.
##                                            CNN Resnet18 
In this project, a convolutional neural network based on the ResNet18 architecture was implemented to classify CT images into three categories: non-stroke, hemorrhage, and ischemic stroke. The dataset consisted of 3000 images in total, with 1000 images per class. The data was split into 80% for training and 20% for testing. Images were preprocessed by resizing to 512x512 pixels and normalizing pixel values. Transfer learning was applied using pretrained ImageNet weights. The model was fine-tuned and trained for 10 epochs using the Adam optimizer and cross-entropy loss. Evaluation metrics included accuracy, weighted F1-score, confusion matrix, and ROC curve analysis.


