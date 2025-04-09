
# Week 11 - Loan Eligibility Prediction

This project uses a logistic regression model to predict whether a loan applicant is eligible for a loan.  
The model is trained on a dataset containing information like age, income, credit score, and more.

---

## Project Structure

- `data/credit.csv` – the dataset containing applicant information  
- `utils/preprocessing.py` – handles data loading and preprocessing  
- `models/model_lr.py` – trains and evaluates the logistic regression model  
- `main.py` – runs the full prediction pipeline  
- `README.md` – this project explanation and instructions

---

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python main.py
```

The script will:
- Load and preprocess the data
- Train the logistic regression model
- Evaluate the model performance (accuracy, confusion matrix, report)

---

## Model Performance

The logistic regression model achieved the following results:

- Accuracy: 0.96
- Confusion Matrix:
```
[[72  1]
 [ 3 24]]
```

- Classification Report:
```
              precision    recall  f1-score   support
           0       0.96      0.99      0.97        73
           1       0.96      0.89      0.92        27
    accuracy                           0.96       100
   macro avg       0.96      0.94      0.95       100
weighted avg       0.96      0.96      0.96       100
```

---

## Logging

Logging is used in this project to show steps like loading data, splitting the dataset, training, and evaluating the model.

---

## Author

- Name: Luyu
- Student number:040986748 

