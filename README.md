# Graduate-admission-outcomes-forecasting - Reserach Project



## Table of Contents
- [Introduction](#introduction)
- [Goals of the Project](#goals-of-the-project)
- [Key Problems Addressed](#key-problems-addressed)
- [Project Setup](#project-setup)
- [Data and Preprocessing](#data-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
  - [Random Forest](#random-forest)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Recurrent Neural Network (RNN)](#recurrent-neural-network-rnn)
- [Results](#results)
- [Explainable AI Techniques](#explainable-ai-techniques)
- [Fairness Techniques](#fairness-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Future Work](#future-work)
- [Challenges and Limitations](#challenges-and-limitations)
- [Ethical Considerations](#ethical-considerations)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict graduate admission outcomes for universities in Dhaka, Bangladesh, using a dataset of 400 student records sourced from Yocket. The goal is to develop a robust predictive model that assists students in understanding their chances of admission based on various academic metrics.

We employed machine learning algorithms such as Random Forest, Recurrent Neural Networks (RNN), Support Vector Machines (SVM), Logistic Regression, and XGBoost. To ensure transparency, we used Explainable AI techniques like LIME (Local Interpretable Model-Agnostic Explanations) to interpret model predictions.

## Goals of the Project
- **Predictive Accuracy:** Achieve high accuracy in predicting admission outcomes.
- **Interpretability:** Provide insights into the factors influencing admission decisions.
- **Fairness:** Ensure that model predictions are fair and unbiased across different demographic groups.

## Key Problems Addressed
- **Data Scarcity:** Compiling data from Yocket to address the lack of comprehensive datasets for graduate admissions in Bangladesh.
- **Model Interpretability:** Providing insights using LIME to explain model predictions.
- **Bias in Admissions:** Identifying and mitigating biases related to gender, ethnicity, or socioeconomic status.
- **Complexity of Admission Processes:** Capturing multiple influencing factors like academic performance, test scores, and personal attributes.

## Project Setup

### Clone the Repository
```bash
git clone https://github.com/yourusername/graduation-admission-prediction.git
cd graduation-admission-prediction
```

## Install Required Packages
Ensure Python 3.7+ is installed. To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```
## Directory Structure
```bash
├── data             # Contains datasets
├── notebooks        # Jupyter notebooks for exploration
├── src              # Source code for the models
├── requirements.txt # Package dependencies
├── README.md        # Project documentation
```

# Data and Preprocessing

## Dataset Overview
The dataset is named `trial.csv` and contains the following features:

- **GRE Score**
- **Toefl Score**
- **Department**
- **CGPA**
- **Research Papers**
- **Projects**
- **Work Experience**
- **University Ranking**
- **Admission Status (Target Variable)**

## Preprocessing Steps

1. **Handling Missing Values**: Fill missing values using forward fill.
2. **Encoding Categorical Variables**: Use one-hot encoding for categorical variables like `Department`.
3. **Feature Scaling**: Normalize features to ensure consistent scaling across the dataset.





