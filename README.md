# Predicting Student Success and Dropout Rates: A Machine Learning Approach

This repository contains the implementation and analysis for the dissertation: **"Predicting Student Success and Dropout Rates: A Machine Learning Approach with LIME for Interpretability and Bias Detection."**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Key Results](#key-results)
7. [Dataset](#dataset)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction
This project investigates how machine learning models, such as Logistic Regression and Random Forest, can predict student outcomes (graduation, dropout, enrollment) in higher education. It integrates LIME (Local Interpretable Model-Agnostic Explanations) to enhance model interpretability and address potential biases.

---

## Features
- **Predictive Modeling:** Logistic Regression and Random Forest for classification tasks.
- **Interpretability:** LIME explanations for transparency in decision-making.
- **Bias Detection:** Identifies and mitigates biases in demographic data.
- **Comprehensive Analysis:** Evaluation using metrics like accuracy, precision, recall, F1-score, and AUC.

---

## Installation
1. Clone the repository:
   git clone https://github.com/username/ml-student-success-prediction.git

2. Navigate to the project direxctory:
   cd ml-student-success-prediction

3. Install dependencies
   pip install -r requirements.txt

## Usage 
**Data Preprocessing:** Prepare the dataset using the data_preprocessing.py script.
**Model Training:** Train models with train.py:
                 python train.py --model logistic_regression
                 python train.py --model random_forest
**Prediction:** Use predict.py to make predictions:
               python predict.py --input student_data.csv
**Interpretability:** Generate LIME explanations with lime_analysis.py

## Methodology
**Data Collection:** Data from the UCI Machine Learning Repository, featuring demographic, academic, and socio-economic variables.
**Modeling:** Logistic Regression for simplicity and Random Forest for handling complex interactions.
**Evaluation Metrics:** Accuracy, precision, recall, F1-score, and AUC for model comparison.
**Interpretability:** LIME used to explain individual predictions and identify biases.

## Key Results
Logistic Regression achieved an accuracy of 92% and an AUC of 0.957.
Random Forest achieved an accuracy of 90% and an AUC of 0.952.
LIME identified potential biases in demographic features such as Nationality and Marital Status.

## Dataset
**Source:** UCI Machine Learning Repository.
**Features:** 37 features including demographic, academic, and socio-economic data.
**Target Variable:** Graduate (1), Dropout (0), Enrolled (dropped for binary classification).

## Limitations and Future Work
**Bias Mitigation:** Use SHAP for global interpretability.  
**Scalability:** Optimize LIME for large datasets.  
**Dynamic Updates:** Enable real-time model updates for current predictions.  
**Model Diversity:** Test advanced algorithms like XGBoost or Neural Networks.  
**Additional Features:** Integrate data like extracurriculars and attendance.  
## Contributing
Contributions are welcome! Fork the repo, create a branch, commit your changes, and open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
