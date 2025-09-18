# Mlops
# Wine Classification Project

## Problem Statement

The goal of this project is to classify wine samples into 3 categories using their chemical properties. We train and compare multiple machine learning models and track experiments using MLflow.

## Dataset

* Wine dataset from `sklearn.datasets`
* 178 wine samples
* 13 chemical features (alcohol, malic acid, ash, alcalinity, magnesium, flavanoids, etc.)
* Target: 3 wine classes

## Model Selection and Results

Three models were trained and evaluated on accuracy, precision, recall, and F1-score:

* Logistic Regression
  Accuracy: 0.9815, Precision: 0.9825, Recall: 0.9815, F1-score: 0.9815
* Random Forest
  Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1-score: 1.0000
* SVM
  Accuracy: 0.9815, Precision: 0.9823, Recall: 0.9815, F1-score: 0.9814

Observation: Random Forest performed the best with perfect accuracy.

## MLflow Logging

* Logged model parameters, metrics, and confusion matrices

## Model Registration

* The best model (Random Forest) was registered in the MLflow Model Registry as `Wine_Classifier`.
* <img width="1651" height="685" alt="image" src="https://github.com/user-attachments/assets/b6c806e1-7af5-4e9d-825b-34fe0f502c62" />
<img width="1655" height="692" alt="image" src="https://github.com/user-attachments/assets/00aa0310-c7d5-47c0-aa47-e1cc7ce3624e" />



## Instructions to Run the Code

1. Install required packages: pandas, scikit-learn, matplotlib, seaborn, joblib, mlflow
2. Run training script: python train\_models.py
3. Log models and metrics: python mlflow\_logging.py
4. Register the best model: python register\_model.py
5. Open MLflow UI: mlflow ui (visit [http://localhost:5000](http://localhost:5000))

