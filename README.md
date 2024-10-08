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
```bash
import pandas as pd

# Load the dataset
data = pd.read_csv('data/trial.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Department'], drop_first=True)

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop('Admission Status', axis=1))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])
data_scaled['Admission Status'] = data['Admission Status']

```
Exploratory Data Analysis : To visualize the distribution of selected features, the following code generates histograms:

```bash
import matplotlib.pyplot as plt

selected_features = ['GRE Score', 'Toefl Score', 'CGPA', 'Research Papers', 'Projects', 'Work Experience', 'University Ranking']

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Create subplots for histograms
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
axes = axes.flatten()

# Plot histograms for selected features
for i, feature in enumerate(selected_features):
    data[feature].plot(kind='hist', ax=axes[i], bins=20, edgecolor='black', color='skyblue')
    axes[i].set_title(f'Histogram of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

```
# Model Training

## Random Forest
We implemented the Random Forest model as follows:
```bash
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split the dataset
X = data_scaled.drop('Admission Status', axis=1)
y = data_scaled['Admission Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_rf))

```
## Support Vector Machine (SVM)

```bash
from sklearn.svm import SVC

# Train the SVM model
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train, y_train)

# Make predictions
y_pred_svm = model_svm.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred_svm))

```
## Recurrent Neural Network (RNN)

```bash
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import LabelBinarizer

# Prepare data for RNN
X_rnn = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
y_rnn = LabelBinarizer().fit_transform(y_train)

# Define the RNN model
model_rnn = Sequential()
model_rnn.add(SimpleRNN(50, activation='relu', input_shape=(X_rnn.shape[1], 1)))
model_rnn.add(Dense(1, activation='sigmoid'))

# Compile the model
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_rnn.fit(X_rnn, y_rnn, epochs=50, batch_size=32)

# Prepare the test data
X_test_rnn = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
y_pred_rnn = model_rnn.predict(X_test_rnn)
y_pred_rnn = (y_pred_rnn > 0.5).astype(int)

# Evaluate the RNN model
print(classification_report(y_test, y_pred_rnn))

```
# Model Training

## Random Forest
We implemented the Random Forest model as follows:

## Results

The following evaluation metrics were obtained for the Random Forest and SVM models:

### Random Forest
- **Accuracy:** 85%
- **Precision:** 0.82
- **Recall:** 0.79
- **F1 Score:** 0.80

### Support Vector Machine
- **Accuracy:** 83%
- **Precision:** 0.80
- **Recall:** 0.77
- **F1 Score:** 0.78

Both models performed well in predicting the admission status of students based on their academic metrics.

# Explainable AI Techniques

To ensure model interpretability, we used the LIME library to explain model predictions:
```bash
import lime
from lime.lime_tabular import LimeTabularExplainer

# Create LIME explainer
explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['Not Admitted', 'Admitted'], mode='classification')

# Explain a prediction
i = 0  # index of the instance to explain
exp = explainer.explain_instance(X_test.values[i], model_rf.predict_proba)

# Visualize the explanation
exp.show_in_notebook(show_table=True)

```
# Fairness Techniques

To ensure fairness in our model predictions, we implemented bias detection using the AI Fairness 360 toolkit. This toolkit allows us to evaluate and mitigate bias in machine learning models.

```bash
from aif360.metrics import BinaryLabelDatasetMetric

# Create a binary label dataset
dataset = BinaryLabelDatasetMetric(data_scaled, label_names=['Admission Status'], protected_attribute_names=['Department'])

# Calculate fairness metrics
print(f'Disparate Impact: {dataset.disparate_impact()}')
print(f'Average Odds Difference: {dataset.average_odds_difference()}')

```
## Evaluation Metrics

We evaluated the models using accuracy, precision, recall, and F1 score to gauge their performance effectively.

## Usage

To run the code, follow these steps:

1. Clone the repository.
2. Install the required packages.
3. Run the Jupyter notebooks in the notebooks directory to explore data analysis and model training.

## Future Work

- Explore advanced deep learning architectures for improved accuracy.
- Investigate transfer learning techniques to leverage pre-trained models.
- Enhance the fairness evaluation using additional metrics and methodologies.

## Challenges and Limitations

- The dataset is limited in size, which may affect the model's generalizability.
- Addressing bias in machine learning models is an ongoing challenge that requires continuous monitoring.

## Ethical Considerations

We are committed to ensuring that our predictive models do not reinforce existing biases in graduate admissions and strive for fairness and transparency in AI.

## References

- LIME
- AI Fairness 360

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for details.








