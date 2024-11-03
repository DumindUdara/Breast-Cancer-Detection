# Breast Cancer Detection

### About

The **Breast Cancer Detection** project utilizes machine learning algorithms to classify tumors as either malignant or benign using the he below mention data set from kaggle. This project aims to assist in early cancer detection by analyzing tumor characteristics.

The code is built using Python libraries, including **pandas** and **NumPy** for data handling, **Seaborn** and **Matplotlib** for visualization, and **scikit-learn** for machine learning algorithms. The main algorithms applied in this project are **Logistic Regression**, **Linear Discriminant Analysis (LDA)**, **Quadratic Discriminant Analysis (QDA)**, **K-Nearest Neighbors (KNN)**, **Random Forest**, and **K-Means Clustering**. These models help evaluate the dataset’s features, aiming to accurately predict the tumor status.

![GitHub Repo stars](https://img.shields.io/github/stars/yourusername/your-repo-name?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/your-repo-name)
![GitHub issues](https://img.shields.io/github/issues/yourusername/your-repo-name)
![GitHub forks](https://img.shields.io/github/forks/yourusername/your-repo-name?style=social)
![Python](https://img.shields.io/badge/Python-3.8-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange)
![Data Source](https://img.shields.io/badge/dataset-Kaggle-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)
<hr>

## Dataset

### Dataset Name
**Breast Cancer Wisconsin (Diagnostic) Data Set**

### Dataset Details
The **Breast Cancer Wisconsin (Diagnostic) Data Set** is a well-known dataset used for the classification of breast cancer tumors. It consists of 569 instances, each representing a different patient, with 30 features that describe various attributes of the tumors. The dataset is divided into two classes:

- **Malignant (1)**: Tumors that are cancerous.
- **Benign (0)**: Tumors that are non-cancerous.

#### Features
The features in the dataset include various measurements computed from digitized images of fine needle aspirate (FNA) of breast masses. They describe characteristics such as:

- **Radius**: Mean of distances from the center to points on the perimeter.
- **Texture**: Standard deviation of gray-scale values.
- **Perimeter**: Mean size of the cancer tumor.
- **Area**: Mean area of the tumor.
- **Smoothness**: Mean of local variations in radius lengths.
- **Compactness**: Mean of (perimeter^2 / area - 1.0).
- **Concavity**: Mean of the severity of concave portions of the contour.
- **Concave Points**: Mean number of concave portions of the contour.
- **Symmetry**: Mean of symmetry values.
- **Fractal Dimension**: Mean of the coastline length of the tumor.

This dataset serves as a benchmark for machine learning algorithms and is widely used for testing classification techniques in medical diagnosis.

### Dataset Link
You can access the dataset from Kaggle at the following link: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

<br><hr>

## Algorithms

### 1. Logistic Regression
Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable. In this project, it helps in predicting whether a tumor is malignant or benign based on various features extracted from the dataset. It estimates the probabilities of the classes (0 for benign and 1 for malignant) and provides a clear interpretation of the relationship between the input features and the target outcome.

### 2. Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis is a classification method that projects features in a way that maximizes the distance between multiple classes while minimizing the variance within each class. It is particularly effective when the classes are well-separated and can help improve classification performance by reducing the dimensionality of the data.

### 3. Quadratic Discriminant Analysis (QDA)
Quadratic Discriminant Analysis is similar to LDA but allows for quadratic boundaries between classes. This means that it can capture more complex relationships in the data when the distribution of the classes is not identical. QDA is useful in scenarios where the assumption of equal covariance matrices for the classes does not hold.

### 4. K-Nearest Neighbors (KNN)
K-Nearest Neighbors is a non-parametric algorithm used for classification and regression. It classifies a data point based on how its neighbors are classified. The algorithm considers the ‘k’ closest training examples in the feature space. It’s simple to implement and often yields good results, especially in cases where the decision boundary is irregular.

### 5. Random Forest
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their predictions (for classification) or the mean prediction (for regression). This method improves the model's accuracy and controls overfitting by averaging the results of multiple trees, making it robust against noise and outliers in the dataset.

### 6. K-Means Clustering
K-Means Clustering is an unsupervised learning algorithm used for partitioning the dataset into K distinct clusters. In the context of this project, K-Means can help identify natural groupings within the data, such as clustering different types of tumors based on their features. It can provide insights into patterns in the data that may not be immediately obvious.
<br>

## Conclusion

In this project, we employed multiple machine learning algorithms to predict the likelihood of breast cancer based on the Breast Cancer Wisconsin (Diagnostic) dataset. The algorithms tested included Logistic Regression, K-Neighbors Classifier, Random Forest Classifier, Linear Discriminant Analysis, and Quadratic Discriminant Analysis. 

The prediction accuracy for each classifier is as follows:

- **Logistic Regression**: 96.49%
- **K-Neighbors Classifier**: 94.74%
- **Random Forest Classifier**: 95.61%
- **Linear Discriminant Analysis**: 96.49%
- **Quadratic Discriminant Analysis**: 95.61%

Overall, Logistic Regression and Linear Discriminant Analysis achieved the highest accuracy, both at 96.49%, demonstrating their effectiveness in classifying tumors as malignant or benign. The K-Neighbors Classifier and Random Forest Classifier also performed well, achieving accuracies above 94%, indicating that these models are reliable options for this classification task.

These results suggest that the chosen algorithms are capable of making accurate predictions in breast cancer diagnosis, highlighting the potential of machine learning in supporting medical decision-making. Future work could involve tuning hyperparameters, exploring additional classifiers, and incorporating more features to further enhance model performance.
