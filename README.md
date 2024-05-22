# Online Payment Fraud Detection using Machine Learning

## Overview

This project focuses on detecting online payment fraud using machine learning techniques. With a dataset of over 6 million transactions, the model aims to accurately classify transactions as fraudulent or non-fraudulent. The standout model, a Decision Tree Classifier, demonstrates high accuracy and precision, ensuring robust fraud detection.

## Dataset

- **Number of variables:** 5
- **Number of observations:** 6,362,620
- **Missing cells:** 0
- **Duplicate rows:** 101,134 (1.6%)

### Variable Types

- **Categorical:** 2 (type, isFraud)
- **Numeric:** 3 (amount, oldbalanceOrg, newbalanceOrig)

## Project Structure

```
|-- data/
|   |-- Dataa.csv         # Dataset file
|-- notebooks/
|   |-- Data_Profiling.ipynb  # Notebook for data profiling
|   |-- Model_Training.ipynb  # Notebook for model training and evaluation
|-- src/
|   |-- data_preprocessing.py # Script for data preprocessing
|   |-- model_training.py     # Script for training models
|-- README.md               # Project documentation
|-- requirements.txt        # Required Python packages
```

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/SuryanshuVerma/online-payment-fraud-detection.git
    cd online-payment-fraud-detection
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Profiling

Generate a detailed report of the dataset using Pandas Profiling:

```python
from ydata_profiling import ProfileReport
import pandas as pd

# Load dataset
dataset = pd.read_csv('data/Dataa.csv')

# Generate report
profile = ProfileReport(dataset, title='Pandas Profiling Report', explorative=True)
profile.to_file("data_profiling_report.html")
```

### Data Preprocessing

Preprocess the data for model training:

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load dataset
dataset = pd.read_csv('data/Dataa.csv')

# Drop unnecessary columns
dataset.drop(columns=["nameDest", "isFlaggedFraud", "oldbalanceDest", "newbalanceDest"], inplace=True)

# Encode categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(dataset.iloc[:, :-1].values)
y = dataset.iloc[:, -1].values
```

### Model Training

Train the Decision Tree Classifier and Logistic Regression models:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
LRC = LogisticRegression(random_state=0)
LRC.fit(X_train, y_train)

DTC = DecisionTreeClassifier(criterion="entropy", max_depth=25, random_state=42)
DTC.fit(X_train, y_train)
```

### Evaluation

Evaluate the models using accuracy and F1 score:

```python
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Logistic Regression Evaluation
y_pred_LRC = LRC.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_LRC))
print("Logistic Regression F1 Score:", f1_score(y_test, y_pred_LRC))

# Decision Tree Classifier Evaluation
y_pred_DTC = DTC.predict(X_test)
print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_DTC))
print("Decision Tree Classifier F1 Score:", f1_score(y_test, y_pred_DTC))
```

### Prediction Example

Use the trained Decision Tree Classifier to predict whether a transaction is fraudulent or not. 

Here is an example prediction:

```python
import numpy as np

# Example features for a transaction
# Format: [type, amount, oldbalanceOrg, newbalanceOrig]
# Categorical variables should be one-hot encoded
features = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 9000.60, 9000.60, 0.0]])

# Predict using the trained Decision Tree Classifier
prediction = DTC.predict(features)
print("Predicted value:", prediction)

# Interpretation
if prediction == 0:
    print("The transaction is not fraudulent.")
else:
    print("The transaction is fraudulent.")
```

In this example, the predicted value is `0`, indicating that the transaction is not fraudulent.

## Conclusion

In the pursuit of identifying online fraud, the Decision Tree Classifier emerged as the standout choice. Its high F1 score and impressive accuracy underscore its effectiveness in distinguishing fraudulent transactions. This selection reflects a commitment to excellence and precision, ensuring the protection of digital integrity with sophistication and poise.

