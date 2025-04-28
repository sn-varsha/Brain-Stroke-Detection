# Stroke_Detection-and-Segmentation-by-Using-Machine-Learning

*In this project, a convolutional neural network (CNN) based on ResNet18 architecture was developed to classify stroke CT images into three categories: non-stroke, hemorrhage, and ischemic. In addition, classical machine learning techniques such as U-Net were used for image segmentation tasks. The project includes preprocessing, training, and evaluation stages with accuracy, weighted F1-score, confusion matrix, and ROC curve analyses. The dataset was obtained from Türkiye Ministry of Health Open Data Portal [https://acikveri.saglik.gov.tr/Home/DataSets?categoryId=10].*

##                                          CNN Resnet18 Results
In this project, a convolutional neural network based on the ResNet18 architecture was implemented to classify CT images into three categories: non-stroke, hemorrhage, and ischemic stroke. The dataset consisted of 3000 images in total, with 1000 images per class. The data was split into 80% for training and 20% for testing. Images were preprocessed by resizing to 512x512 pixels and normalizing pixel values. Transfer learning was applied using pretrained ImageNet weights. The model was fine-tuned and trained for 10 epochs using the Adam optimizer and cross-entropy loss. Evaluation metrics included accuracy, weighted F1-score, confusion matrix, and ROC curve analysis.


# ![roc_curve](https://github.com/user-attachments/assets/fe86f650-84ff-439b-b506-e5673e457d0c)
# ![Confusion_Matrix](https://github.com/user-attachments/assets/47449230-0f4c-4357-8838-ead9e5231d70)
# ![Test_Sonuçları](https://github.com/user-attachments/assets/a3b3e1ea-a238-4799-a7be-22eafbcada59)

