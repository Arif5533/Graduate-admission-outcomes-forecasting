# Graduate-admission-outcomes-forecasting - Reserach Project

README.md for Graduate Admission Prediction Project
Table of Contents
Introduction
Key Problems Addressed
Project Setup
Complete Project Code
Data and Preprocessing
Model Training
Fairness Techniques
Evaluation Metrics
Usage
Results
Example Notebooks
Future Work
Challenges and Limitations
Ethical Considerations
References
Contributing
License
Introduction
This project aims to predict graduate admission outcomes for universities in Dhaka, Bangladesh, using a dataset of 400 student records sourced from Yocket. The primary goal is to develop a robust predictive model that can assist students in understanding their chances of admission based on various academic metrics. We employed several machine learning algorithms, including Random Forest, Recurrent Neural Networks (RNN), Support Vector Machines (SVM), Logistic Regression, and XGBoost. To enhance transparency and trust in our models, we implemented Explainable AI techniques, specifically LIME (Local Interpretable Model-Agnostic Explanations), to interpret model predictions.
Goals of the Project
Predictive Accuracy: Achieve a high level of accuracy in predicting admission outcomes.
Interpretability: Provide insights into the factors influencing admission decisions.
Fairness: Ensure that the model predictions are fair and unbiased across different demographic groups.
Key Problems Addressed
Data Scarcity: The lack of comprehensive datasets for graduate admissions in Bangladesh poses a significant challenge. By compiling data from Yocket, we aim to bridge this gap.
Model Interpretability: Machine learning models are often seen as "black boxes." By using LIME, we provide insights into how models make predictions, which is crucial for stakeholders such as students and educational institutions.
Bias in Admissions: Admission processes can be influenced by biases related to gender, ethnicity, or socioeconomic status. Our project seeks to identify and mitigate these biases in our predictive models.
Complexity of Admission Processes: Graduate admissions are affected by various factors including academic performance, standardized test scores, and personal attributes. Capturing this complexity is essential for accurate predictions.
Project Setup
To set up the project locally, follow these steps:
Clone the Repository
bash
git clone https://github.com/yourusername/graduation-admission-prediction.git
cd graduation-admission-prediction

Install Required Packages
Ensure you have Python installed (preferably version 3.7 or higher). Then install the required libraries:
bash
pip install -r requirements.txt

Directory Structure
The project follows a structured directory layout:
text
├── data             # Contains datasets
├── notebooks        # Jupyter notebooks for exploration
├── src              # Source code for the models
├── requirements.txt  # Package dependencies
├── README.md        # Project documentation

Complete Project Code
Data Loading and Preprocessing
The following code snippets illustrate how to load and preprocess the data:
python
import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the dataset by handling missing values and encoding categorical variables."""
    # Handling missing values
    data.fillna(method='ffill', inplace=True)

    # One-hot encoding for categorical variables
    categorical_cols = ['gender', 'undergraduate_institution']
    data = pd.get_dummies(data, columns=categorical_cols)

    return data

data = load_data('data/student_records.csv')
processed_data = preprocess_data(data)

Model Training Example
Here’s an example of how to train a Random Forest model:
python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = processed_data.drop('admission_status', axis=1)  # Features
y = processed_data['admission_status']                 # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize model with 100 trees
model.fit(X_train, y_train)  # Train the model

accuracy = model.score(X_test, y_test)  # Evaluate accuracy on test set
print(f'Accuracy: {accuracy:.2f}')

LIME for Model Interpretability
Using LIME to explain predictions:
python
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                     feature_names=X.columns,
                                                     class_names=['Not Admitted', 'Admitted'],
                                                     mode='classification')

exp = explainer.explain_instance(X_test.values[0], model.predict_proba)
exp.show_in_notebook()

Data and Preprocessing
Dataset Overview
The dataset consists of the following features:
GRE Score: Graduate Record Examination score.
TOEFL Score: Test of English as a Foreign Language score.
Undergraduate GPA: Grade Point Average from undergraduate studies.
Work Experience: Number of years of relevant work experience.
Recommendation Letters: Number of recommendation letters submitted.
Admission Status: Target variable indicating whether the student was admitted.
Preprocessing Steps
Handling Missing Values: We used forward fill to handle missing values.
Encoding Categorical Variables: Categorical variables were converted into numerical format using one-hot encoding.
Feature Scaling: Numerical features were normalized to improve model performance.
Exploratory Data Analysis (EDA)
Before diving into modeling, we performed EDA to understand the dataset better:
python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(x='admission_status', data=data)
plt.title('Distribution of Admission Status')
plt.show()

This visualization helps us understand the balance between admitted and not admitted students.
Model Training
Overview of Models Used
Random Forest: Combines multiple decision trees to improve accuracy and control overfitting.
RNN: Suitable for sequential data but applied here for its ability to capture complex relationships.
SVM: Effective in high-dimensional spaces; used with different kernels for experimentation.
Logistic Regression: A baseline linear model providing insights into feature importance.
XGBoost: An optimized implementation of gradient boosting that performs well on structured data.
Training Process
Each model was trained using a similar approach:
Split the dataset into training and testing sets.
Train the model on the training set.
Evaluate performance on the testing set.
Hyperparameter Tuning
Hyperparameter tuning was performed using Grid Search or Random Search methods to find optimal parameters for each model.
python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f'Best Parameters: {grid_search.best_params_}')

Fairness Techniques
Bias Detection Methods
To ensure fairness in our predictions:
Statistical Parity Difference: Measures whether different demographic groups receive similar prediction rates.
Equal Opportunity Difference: Examines whether true positive rates are similar across groups.
Mitigation Strategies
We applied various strategies to mitigate identified biases:
Re-weighting Techniques: Instances were re-weighted based on their demographic attributes to reduce bias during training.
Adversarial Debiasing: We explored adversarial debiasing techniques where an adversary tries to predict sensitive attributes from model predictions.
Evaluation Metrics
Metrics Overview
The evaluation metrics used include:
Accuracy: The ratio of correctly predicted instances to total instances.
F1 Score: The harmonic mean of precision and recall; useful for imbalanced datasets.
Confusion Matrix: Visual representation of true vs predicted classifications.
python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

Visualizing Confusion Matrix
python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Admitted', 'Admitted'], yticklabels=['Not Admitted', 'Admitted'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

Usage
To use this project:
Place your dataset in the data directory.
Modify the load_data function if necessary to fit your dataset's structure.
Execute Jupyter notebooks located in the notebooks directory for exploratory data analysis and model training.
Running Jupyter Notebooks
To run Jupyter notebooks:
bash
jupyter notebook notebooks/

This command will open Jupyter Notebook in your default web browser.
Results
Performance Summary
The performance results indicate that:
The Random Forest model achieved an accuracy of approximately 85%.
XGBoost followed closely with an accuracy of around 83%.
The interpretability provided by LIME revealed that GRE scores and undergraduate GPA were significant predictors of admission outcomes.
Example Results Visualization
python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.barh(X.columns[:-1], model.feature_importances_)
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest Model')
plt.show()

Example Notebooks
Several Jupyter notebooks are included for demonstration purposes:
Exploratory_Data_Analysis.ipynb: Contains visualizations and initial data exploration techniques such as histograms and box plots.
Model_Training.ipynb: Step-by-step guide on training various models with hyperparameter tuning included.
Model_Interpretation.ipynb: Demonstrates how to use LIME for interpreting predictions made by different models.
Future Work
This project lays the groundwork for future research in graduate admission prediction by:
Expanding the dataset with more records from various universities across Bangladesh.
Integrating additional features such as personal statements or extracurricular activities.
Exploring advanced deep learning techniques like Convolutional Neural Networks (CNNs) or Transformers for improved prediction accuracy.
Implementing real-time prediction capabilities through a web application interface.
Challenges and Limitations
Dataset Limitations
While this project has made significant strides in predicting graduate admissions:
The limited size of the dataset may restrict generalizability across different institutions or regions.
Potential biases inherent in historical admission data may still affect predictions despite mitigation efforts.
Model Complexity
Some models like RNNs may require extensive computational resources and fine-tuning which can be challenging without adequate infrastructure.
Interpretability Challenges
While LIME provides insights into decision-making processes, interpreting complex models remains a challenge when deploying advanced techniques without adequate explainability tools.
Ethical Considerations
Responsible AI Use
As we develop predictive models that influence educational opportunities:
It's crucial to ensure that our models do not perpetuate existing biases or inequalities present in historical data.
Transparency about how predictions are made should be maintained so stakeholders can understand potential limitations or biases in recommendations.
Data Privacy
Ensure that all personal data used is anonymized and handled according to ethical guidelines regarding privacy and consent.
References
Yocket - Source of student records used in this project.
LIME Documentation - For understanding Local Interpretable Model-Agnostic Explanations (LIME).
Scikit-learn Documentation - For machine learning algorithms used throughout this project.
Contributing
Contributions are welcome! Please follow these steps to contribute:
Fork the repository on GitHub.
Create a new branch (git checkout -b feature/YourFeature).
Make your changes and commit them (git commit -m 'Add some feature').
Push your branch (git push origin feature/YourFeature).
Open a pull request detailing your changes.
License
This project is licensed under the MIT License - see the LICENSE file for details.
