# Cancer classification using CNN
Group project for the uni course _"Deep Learning for Neuroscience and Rehabilitation"_. The aim was to improve the results obtained in [this notebook](https://www.kaggle.com/code/aravindbn/cancer-classification-using-cnn-93/notebook) using the knowledge learned during the course.

## 1. Introduction
This project aimed to enhance the performance of Convolutional Neural Networks (CNNs) for image classification tasks by applying techniques such as stain normalization and data augmentation. The primary goal was to improve accuracy, sensitivity, and specificity while addressing issues related to class imbalance.

## 2. Methodology

### 2.1. Data Preparation
The dataset consisted of images that underwent preprocessing, including stain normalization to reduce variability in color representation across different samples.

### 2.2. CNN Model Structure
The CNN architecture utilized a transfer learning approach with a pre-trained backbone (DenseNet201). The model was constructed with several convolutional layers, followed by pooling, dropout, and batch normalization layers. The final layer utilized a softmax activation function for classification.

### 2.3. Class Weight Calculation
Class weights were computed to address class imbalance in the dataset. This ensured that the model paid equal attention to underrepresented classes during training.

### 2.4. Data Augmentation
Data augmentation techniques were applied to both the training and validation datasets to increase the diversity of the training samples. This included geometric transformations and color variations to enhance generalization.

### 2.5. Model Training and Evaluation
The model was trained with various configurations, each corresponding to different combinations of data augmentation and stain normalization. The training involved early stopping to prevent overfitting.

### 2.6. Evaluation Metrics
The models were evaluated using metrics such as accuracy, sensitivity, and specificity to assess their performance on unseen test data.

## 3. Results

The following table summarizes the results obtained from different configurations of the CNN models:

| ID CNN | Train Acc | Test Acc | Sensitivity | Specificity |
|--------|-----------|----------|-------------|--------------|
| 001    | 0.997     | 0.916    | 0.93        | 0.88         |
| 002    | 0.91      | 0.927    | 0.95        | 0.87         |
| 003    | 0.991     | 0.85     | 0.93        | 0.68         |
| 004    | 0.81      | 0.807    | 0.78        | 0.86         |
| 005    | 0.671     | 0.732    | 0.96        | 0.24         |

### 3.1. Observations
- **Model 001** achieved the highest training accuracy (99.7%) and a strong test accuracy of 91.6%, indicating excellent generalization.
- **Model 002** also performed well with a test accuracy of 92.7% and high sensitivity (95%), making it suitable for applications requiring high detection rates.
- **Model 003** exhibited a notable drop in test accuracy (85%), despite a high training accuracy (99.1%), indicating potential overfitting.
- **Model 004** and **Model 005** showed lower overall performance, with test accuracies of 80.7% and 73.2%, respectively. Model 005 particularly struggled with specificity, indicating a high false positive rate.

## 4. Conclusion
The project successfully demonstrated the impact of stain normalization and data augmentation on the performance of CNNs in image classification tasks. While some models showed strong performance, further optimization and fine-tuning are necessary to enhance the robustness of the models, particularly for scenarios with class imbalance.

Future work may explore additional augmentation techniques and hyperparameter tuning to improve model performance further.
