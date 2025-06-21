# Breast Cancer Prediction Project

This project focuses on building and evaluating machine learning models to predict breast cancer diagnosis (Malignant or Benign) using the Breast Cancer Wisconsin (Diagnostic) Dataset. It covers the entire machine learning pipeline, from data loading and exploratory data analysis to model training, evaluation, optimal threshold determination, and finally, creating a production-ready prediction function.

## Table of Contents

1.  [Introduction](https://www.google.com/search?q=%23introduction)
2.  [Dataset](https://www.google.com/search?q=%23dataset)
3.  [Project Structure](https://www.google.com/search?q=%23project-structure)
4.  [Key Steps & Methodology](https://www.google.com/search?q=%23key-steps--methodology)
      * [1. Data Loading and Initial Inspection](https://www.google.com/search?q=%231-data-loading-and-initial-inspection)
      * [2. Exploratory Data Analysis (EDA)](https://www.google.com/search?q=%232-exploratory-data-analysis-eda)
      * [3. Data Preprocessing](https://www.google.com/search?q=%233-data-preprocessing)
      * [4. Model Training and Evaluation](https://www.google.com/search?q=%234-model-training-and-evaluation)
      * [5. Optimal Threshold Determination](https://www.google.com/search?q=%235-optimal-threshold-determination)
      * [6. Production-Ready Prediction Function](https://www.google.com/search?q=%236-production-ready-prediction-function)
5.  [Results](https://www.google.com/search?q=%23results)
6.  [Requirements](https://www.google.com/search?q=%23requirements)
7.  [Usage](https://www.google.com/search?q=%23usage)
8.  [License](https://www.google.com/search?q=%23license)

## Introduction

Early and accurate diagnosis of breast cancer is crucial for effective treatment. This project aims to develop a classification model that can assist in distinguishing between benign and malignant breast masses based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The emphasis is on building a robust model and understanding the impact of different probability thresholds on predictions, especially for critical medical diagnoses where false negatives might have severe consequences.

## Dataset

The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is available through `sklearn.datasets`. This dataset contains 569 instances with 30 features computed from digitized images of breast masses, and a target variable indicating whether the mass is malignant (0) or benign (1).

**Features include:**

  * Radius (mean of distances from center to points on the perimeter)
  * Texture (standard deviation of gray-scale values)
  * Perimeter
  * Area
  * Smoothness (local variation in radius lengths)
  * Compactness (perimeter^2 / area - 1.0)
  * Concavity (severity of concave portions of the contour)
  * Concave points (number of concave portions of the contour)
  * Symmetry
  * Fractal dimension ("coastline approximation" - 1)
  * And their 'worst' or 'standard error' values.

The dataset is slightly imbalanced, with approximately 37% malignant cases and 63% benign cases.

## Project Structure

The core of this project is implemented in a Jupyter Notebook (`Untitled9.ipynb`).

## Key Steps & Methodology

### 1\. Data Loading and Initial Inspection

The dataset is loaded using `load_breast_cancer()` from `sklearn.datasets` and converted into a pandas DataFrame. Initial data inspection involves:

  * Displaying the first few rows (`df.head()`).
  * Checking data types and non-null counts (`df.info()`).
  * Generating descriptive statistics (`df.describe()`).
  * Analyzing the distribution of the target variable (malignant vs. benign).

### 2\. Exploratory Data Analysis (EDA)

Basic EDA includes visualizing the distribution of the target variable using a count plot to understand class balance.

### 3\. Data Preprocessing

  * **Feature-Target Split:** The dataset is separated into features (X) and the target variable (y).
  * **Train-Test Split:** The data is split into training and testing sets (typically 80% training, 20% testing) to evaluate model performance on unseen data.
  * **Feature Scaling:** Features are standardized using `StandardScaler` to ensure that all features contribute equally to the model and to improve the performance of distance-based algorithms.

### 4\. Model Training and Evaluation

Several common classification algorithms are trained and evaluated:

  * Logistic Regression
  * Support Vector Machine (SVC)
  * Random Forest Classifier
  * Gradient Boosting Classifier
  * K-Nearest Neighbors Classifier
  * MLP Classifier (Neural Network)

Models are evaluated using a comprehensive set of metrics, including:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC AUC Score
  * Confusion Matrix

### 5\. Optimal Threshold Determination

Instead of relying solely on the default 0.5 probability threshold for binary classification, an "optimal" threshold is determined. This threshold, identified as **0.3387**, is crucial for applications where minimizing specific types of errors (e.g., false negatives in medical diagnosis) is prioritized. The determination method likely involves analyzing the Receiver Operating Characteristic (ROC) curve and potentially using metrics like Youden's J statistic or maximizing the F1-score to find the best balance between True Positive Rate (Sensitivity) and False Positive Rate.

### 6\. Production-Ready Prediction Function

A flexible `predict_breast_cancer` function is implemented. This function takes:

  * The trained model.
  * The fitted scaler.
  * New patient data.
  * An optional `threshold_malignant` parameter, allowing the user to specify a custom probability threshold for classifying a case as Malignant.

This function is tested with sample data, demonstrating how a custom threshold can change the predicted outcome for patients whose malignant probability falls between the default 0.5 and the custom threshold. For example, a patient with a malignant probability of `0.4813` would be classified as "Benign" with the default 0.5 threshold, but as "Malignant" with the optimized `0.3387` threshold.

## Results

The project successfully identifies the optimal threshold for the given model and dataset, and demonstrates its practical application in a production-ready prediction function. The ability to adjust the threshold allows for fine-tuning the model's sensitivity based on the specific requirements of the application, such as prioritizing the reduction of false negatives in breast cancer diagnosis.

## Requirements

The project uses standard Python libraries for data science and machine learning. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

To run this project:

1.  Ensure you have Jupyter Notebook or JupyterLab installed.
2.  Download the `Breast_Cancer_Prediction.ipynb` file.
3.  Open the notebook in Jupyter and run all cells sequentially.

The notebook will:

  * Load the dataset.
  * Perform EDA.
  * Preprocess the data.
  * Train and evaluate multiple models.
  * Determine and apply the optimal threshold.
  * Demonstrate the `predict_breast_cancer` function.

## License

This project is open-sourced under the MIT License.
