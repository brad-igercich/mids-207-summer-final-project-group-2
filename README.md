# mids-207-summer-final-project-group-2

### Authors: Bradley Igercich, Sahana Sankar, Laura Lubben, Chloe McGlynn

## Baseline Presentation
[iSchool Google Drive Link](https://docs.google.com/presentation/d/151mlOsWOUO0_-WzGq1jwh9FFm3hCE9S9R8waqGN6I_o/edit?usp=sharing)

## Overview of ML for Brain Tumors: Early Detection for Better Outcomes

### **Motivation:**
  * **Question:** Can we train a model to use MRI brain scans to reliably predict the presence of no brain tumor vs. presence of a glioma, meningioma or pituitary tumor?
  * **Importance:**
    * According to the American Brain Tumor Association, approximately 700,000 people in the United States are living with a primary brain tumor, and nearly 80,000 new cases are diagnosed each year.
    * Magnetic Resonance Imaging (MRI) is one of the most effective and non-invasive tools for brain tumor detection. However, the manual analysis of MRI images by radiologists can be time-consuming.
    * Despite advancements in medical technology, the five-year survival rate for patients with malignant brain tumors remains low, at around 36%. This underscores the importance of early detection, which can lead to more effective treatment options, such as surgery, radiation, and chemotherapy.

### **Data:**
* Data Source:
  * [Brain MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)
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
 
### Repository Organization
* [DATASCI 207 Final Project Notebooks (ALL)](DATASCI207_FinalProject/Notebooks)
  *  [Data Pre-Processing](DATASCI207_FinalProject/Notebooks/1_Data_Preprocessing.ipynb)
      * Import Libraries
      * Download and Unzip Dataset from Kaggle
      * Exploratory Data Analysis:
        * Displayed the first five MRI images from each class.
        * Counted and visualized the distribution of images in the training and testing datasets.
        * Check for Duplicate Images:
          * Implemented functions to compute file hashes and remove duplicates based on these hashes.
          * Displayed file counts before and after removing duplicates.
  *  [Baseline Model](DATASCI207_FinalProject/Notebooks/2_Baseline_Model.ipynb)
      * Import Libraries
      * Baseline Model is constructed using build_baseline_model function
        * Model is trained for 20 epochs with early stopping if the validation accuracy doesn't improve for 4 consecutive epochs.
        * Model is evaluated on both the training and validation datasets.
      * MaxPooling vs AveragePooling
        * Both configurations were tested over 10 epochs. The results showed no significant difference in validation accuracy between the two methods.
        * However, due to the nature of MRI images, MaxPooling was chosen for further modeling.
      * Results
        * Validation Accuracy - 
        * Training Accuracy -  
  *  [CNN](DATASCI207_FinalProject/Notebooks/3_CNN.ipynb)
      * Import Libraries
      * Load the pre-processed training, validation, and test datasets
      * 2-Layer CNN is constructed
        * Model is trained for 20 epochs with early stopping.
        * Model is evaluated on both the training and validation datasets.
      * 3-layer CNN is constructed 
        * Model is trained for 20 epochs with early stopping.
        * Model is evaluated on both the training and validation datasets.
      * Experimenting with Filters
        * Experimented with different numbers of filters (12, 24, 48) in the 2-layer CNN model to observe the impact on validation accuracy
      * Results
        * 2-layer CNN model outperformed the 3-layer model, suggesting that increasing the number of layers does not always result in better performance, especially with a smaller dataset
        * Validation Accuracy - 88%
        * Training Accuracy - 99% 
  *  [Hybrid Model](DATASCI207_FinalProject/Notebooks/4_HybridModel.ipynb)
  *  [Test Predictions](DATASCI207_FinalProject/Notebooks/5_TestPredictions.ipynb)
  *  [Multi-class ROC Analysis](DATASCI207_FinalProject/Notebooks/6_Multi_class_ROC_Analysis_final.ipynb)
 
### **Conclusions:**
* The Baseline Model resulted in overfitting to the training data, with a 99% training accuracy and 88% testing accuracy. This model also resulted in lower Recall for 'No Tumor' and 'Meningioma' classes.
* Hybrid Model: Pre-Trained CNN + Transformer Layer performed best with 98% training accuracy and 96% testing accuracy.
* The Hybrid Model improved 'Meningioma' Recall, 'No Tumor' Recall, and F1 Scores for all classes.
* Despite advancements, the Hybrid Model still struggled to correctly predict the presence of 'Meningioma' when compared to other classes, with a Recall of 91% and an F1 score of 93%.
  
### **Contributions:**
  
| Task                  | Bradley | Sahana | Chloe | Laura |
|-----------------------|:-------:|:------:|:-----:|:-----:|
| **Research**          |    x    |   x    |   x   |   x   |
| **Data Cleaning**      |    x    |   x    |   x   |   x   |
| **EDA**               |    x    |   x    |   x   |   x   |
| **Data Preprocessing** |    x    |   x    |   x   |   x   |
| **Prediction Algorithm**|   x    |   x    |   x   |   x   |
| **Model Fine Tuning**  |    x    |   x    |   x   |   x   |
| **Results**           |    x    |   x    |   x   |   x   |
