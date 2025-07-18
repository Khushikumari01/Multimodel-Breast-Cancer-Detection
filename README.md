Breast Cancer Detection using Machine Learning

 Project Overview

This project aims to build a predictive model that can classify whether a tumor is benign or malignant based on features extracted from breast cancer histology images. Using a dataset of diagnostic measurements, we apply machine learning algorithms to help support early detection and diagnosis of breast cancer.

 Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains 30 features (like radius, texture, smoothness, symmetry, etc.) computed from digitized images of breast mass tissue.

* Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
* The dataset contains 569 samples with 30 numeric features each.
* The target label is binary:

  * `M` (Malignant)
  * `B` (Benign)

Tools & Technologies

* Python
* NumPy
* Pandas
* Scikitlearn
* Matplotlib
* Seaborn
* Jupyter Notebook

Workflow

1. Data Loading
   Load the breast cancer dataset from `sklearn.datasets`.

2. Data Preprocessing

   * Convert the dataset to a DataFrame.
   * Encode the labels (`M`/`B`) to numeric values.
   * Normalize the data using `StandardScaler`.

3. Exploratory Data Analysis (EDA)

   * Visualize feature correlations using a heatmap.
   * Check for class balance.

4. Model Training & Evaluation
   Multiple classification models were tested:

   * Logistic Regression
   * Decision Tree Classifier
   * Random Forest Classifier
   * Support Vector Machine (SVM)
   * KNearest Neighbors (KNN)

   Metrics used:

   * Accuracy
   * Confusion Matrix
   * Classification Report (Precision, Recall, F1score)

5. Results
   The bestperforming models achieved high accuracy (above 95%), making them suitable for preliminary medical diagnostics, though further validation is required for clinical use.

 Model Performance

 Model                Accuracy 
    
 Logistic Regression  ~97%    
 Random Forest        ~96%    
 SVM                  ~98%    
 KNN                  ~95%    

Conclusion

This project demonstrates the potential of machine learning in early breast cancer detection. While the models show promising results, further testing with more diverse and realworld data is necessary before clinical deployment.

