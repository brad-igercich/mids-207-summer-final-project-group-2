# mids-207-summer-final-project-group-2

## Baseline Presentation
iSchool Google Drive Link: https://docs.google.com/presentation/d/151mlOsWOUO0_-WzGq1jwh9FFm3hCE9S9R8waqGN6I_o/edit?usp=sharing

## Overview of Final ML Model Project: Identifying Brain Tumors
### **Motivation:**
  * **Question:** Can we train a model to use MRI brain scans to reliably predict the presence of no brain tumor vs. presence of a glioma, meningioma or pituitary tumor?
  * **Importance:** According to the American Brain Tumor Association, approximately 700,000 people in the United States are living with a primary brain tumor, and nearly 80,000 new cases are diagnosed each year.
  Magnetic Resonance Imaging (MRI) is one of the most effective and non-invasive tools for brain tumor detection.
  However, the manual analysis of MRI images by radiologists can be time-consuming.
  Despite advancements in medical technology, the five-year survival rate for patients with malignant brain tumors remains low, at around 36%.
  This underscores the importance of early detection, which can lead to more effective treatment options, such as surgery, radiation, and chemotherapy.

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

### **Modeling:**
* Baseline
  * 2D layers?
* Improvement over Baseline
  * Improvement on the Baseline
  * Try one-hot encoding instead of label encoding; try cross-validation
  * CNN/Transformer Model (combine two different models; Ensemble - multiple algorithms that deliver different answers but identify patterns in consistent answers; are they limited to Decision Forests or can be used with CNNs?)

### **Experiments:**

* Experiments for Pre-Processing portion:
	* Max Pooling 2D vs Average Pooling 2D
	* Input shapes - RGB and Greyscale
	* What do the Training and Test split distributions look like after removing duplicates? Does this address any slight imbalance in the original dataset? (Yes)
	* Step to address model’s slight struggle to differentiate between meningioma and glioma?
	* Alternative Model Experiment:
	  * Binary Classification model to detect tumor v no tumor

### **Conclusions:**

### **Contributions:**
