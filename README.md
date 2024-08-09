# mids-207-summer-final-project-group-2

## Baseline Presentation
iSchool Google Drive Link: https://docs.google.com/presentation/d/151mlOsWOUO0_-WzGq1jwh9FFm3hCE9S9R8waqGN6I_o/edit?usp=sharing

## Overview of Final ML Model Project: Identifying Brain Tumors

### **Motivation:**
  * **Question:** Can we train a model to use MRI brain scans to reliably predict the presence of no brain tumor vs. presence of a glioma, meningioma or pituitary tumor?
  * **Importance:**
    * According to the American Brain Tumor Association, approximately 700,000 people in the United States are living with a primary brain tumor, and nearly 80,000 new cases are diagnosed each year.
    * Magnetic Resonance Imaging (MRI) is one of the most effective and non-invasive tools for brain tumor detection. However, the manual analysis of MRI images by radiologists can be time-consuming.
    * Despite advancements in medical technology, the five-year survival rate for patients with malignant brain tumors remains low, at around 36%. This underscores the importance of early detection, which can lead to more effective treatment options, such as surgery, radiation, and chemotherapy.

### **Data:**
* Data Source:
  * https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data
  * Combined image datasets from figshare.com & other similar kaggle projects
* Data Shape:
  * 7023 human brain MRI images total (including no tumor, meningioma, glioma, pituitary)
  * 1621 images featuring a glioma
  * 1645 images featuring a meningioma
  * 1757 images featuring a pituitary tumor
  * 2000 images featuring no tumor
* Main Features:
  * Greyscale color // color values
  * 224 x 224 pixels // pixel labels
  * Scan orientation

### **Models:**
* Baseline CNN Model
  * Single Conv2D layer (ReLU)
  * MaxPooling2D layer
  * Dropout Rate = 0.3
  * Flatten Layer
  * Dense Layer (softmax)
* 2-layer CNN Model
  * Two Conv2D layers (ReLU)
  * MaxPooling2D layer each
  * Dropout Rate = 0.3 each
  * Flatten Layer
  * Dense Layer (softmax)
* 3-layer CNN Model
  * Three Conv2D layers (ReLU)
  * MaxPooling2D layer each
  * Dropout Rate = 0.3 each
  * Flatten Layer
  * Dense Layer (softmax)
* Hybrid Model: Pre-Trained CNN + Transformer Layer
  * Pre-trained MobileNetV2 Model
  * Reshape for Transformer Layer
  * Multi-head Attention Layer
  * Reshape back to 3D Tensor
  * GlobalAvgPooling
  * Dropout Rate = 0.5
  * Dense Layer (ReLU)
  * Dense Layer (Softmax)

### **Experiments:**
* MaxPooling vs AveragePooling
  * Tested if MaxPooling or AveragePooling layers produced better results when adding complexity to Baseline model.
  * Findings indicated that there was no real difference between two, but we chose to keep the MaxPooling layer to emphasize the pixel values corresponding to tumors in the MRI scans.
* CNN Model Filters
  * Experimented with hyperparameters on 2-layer and 3-layer CNN models as we increased complexity from baseline.
  * Assessed number of filters and whether they had a significant impact on the performance of the models.
  * Generated simple test to evaluate the performance of the following filter counts on overall validation performance: 12, 24, and 48.
  * Findings indicated that more filters performed worse than a lower filter count of 12 and 24, likely due to increasing complexity with a relatively low number of data points. 

### **Conclusions:**
* The Baseline Model resulted in overfitting to the training data, with a 99% training accuracy and 88% testing accuracy. This model also resulted in lower Recall for 'No Tumor' and 'Meningioma' classes.
* Hybrid Model: Pre-Trained CNN + Transformer Layer performed best with 98% training accuracy and 96% testing accuracy.
* The Hybrid Model improved 'Meningioma' Recall, 'No Tumor' Recall, and F1 Scores for all classes.
* Despite advancements, the Hybrid Model still struggled to correctly predict the presence of 'Meningioma' when compared to other classes, with a Recall of 91% and an F1 score of 93%.
  
### **Contributions:**
  
