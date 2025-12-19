# Stress-Analyzer-Assistant

# Stress Analyzer & Assistant

A machine learning–based web application that predicts stress levels among college students and provides personalized, actionable recommendations.

## Features
- Stress level prediction (Low / Medium / High) using a Random Forest classifier
- SHAP-based interpretability for medium and high stress predictions
- Local LLM-powered recommendations using Ollama (LLaMA)
- Student-focused, non-medical guidance

## Tech Stack
- Python, Flask
- Scikit-learn, SHAP
- HTML/CSS
- Ollama (LLaMA 3.1)

## How It Works
1. User enters academic, lifestyle, and mental health–related inputs
2. Model predicts stress level
3. SHAP identifies key contributing factors (for medium/high stress)
4. LLM generates personalized stress-reduction suggestions

## Setup Instructions
```bash
pip install -r requirements.txt
python app.py

## Model Training & Methodology

The stress prediction model was trained using a structured survey-based dataset containing academic, lifestyle, and mental health–related attributes of college students.

### Data Preprocessing
- Categorical features (e.g., Gender, City, Degree, Sleep Duration) were encoded using `LabelEncoder`.
- Numerical features such as CGPA, Academic Pressure, Study Satisfaction, and Work/Study Hours were used as-is.
- The target variable (Stress Level) was encoded into three classes: Low, Medium, and High.

### Class Imbalance Handling
The dataset exhibited class imbalance, particularly for the "Low" stress category.  
To address this:
- SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data.
- This ensured balanced class representation and improved generalization.

### Model Selection
A **Random Forest Classifier** was chosen due to:
- Strong performance on mixed categorical and numerical data
- Robustness to noise and outliers
- Built-in support for feature importance and interpretability

The model was trained with:
- `n_estimators = 200`
- `class_weight = "balanced"`

### Evaluation
The model was evaluated using a held-out test set with metrics including:
- Accuracy
- Precision, Recall, and F1-score

This setup achieved strong performance across all stress categories while maintaining interpretability.

### Interpretability
SHAP (SHapley Additive exPlanations) was used to:
- Identify key contributing factors for Medium and High stress predictions
- Avoid misleading explanations for Low stress cases by design

This ensures explanations are actionable and user-appropriate.
