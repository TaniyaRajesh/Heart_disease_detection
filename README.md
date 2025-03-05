# Heart Disease Prediction using Machine Learning

## Overview
This project predicts heart disease using a Decision Tree classifier. It includes data preprocessing, exploratory data analysis (EDA), model evaluation, and deployment as a web application.

## Dataset
- Features include age, sex, blood pressure, cholesterol, heart rate, and more.
- Target: 0 (No Heart Disease) / 1 (Heart Disease Present).

## Workflow
1. **EDA & Preprocessing**: Outlier removal, feature encoding, and standardization.
2. **Model Training**: Trained a Decision Tree Classifier.
3. **Evaluation**: Assessed performance with classification metrics and a confusion matrix.
4. **Deployment**: Hosted as a web app for real-time predictions.

## Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
```python
import pickle
import numpy as np

with open('dtreeModel.pkl', 'rb') as model_file:
    dtreeModel = pickle.load(model_file)

sample_data = np.array([[0.396, 0.707, -0.967, -0.274, 1.293, -0.391, -1.104, 0.956, -0.707, -0.963, 0.998, 2.033, 1.184]])
prediction = dtreeModel.predict(sample_data)
print('Predicted Class:', prediction)
```

## Web App
Access the deployed web application here: [Heart Disease Prediction Web App](
https://heartdiseasedetection-taniya.streamlit.app/)

## Future Work
- Hyperparameter tuning (GridSearchCV).
- Experiment with Random Forest and Neural Networks.
- Improve web app UI/UX.

## Author
Taniya Rajesh
Developed as part of a machine learning exploration in heart disease prediction.
