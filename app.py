import streamlit as st
import pandas as pd
from models import model_lr
from utils.preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the data
logging.info("Loading and preprocessing the data...")
df = load_data("data/loan_eligibility.csv")

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]


# Split the data into training and testing sets
logging.info("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
logging.info("Training Logistic Regression model...")
model = model_lr.train_model(X_train, y_train)

# Set up the Streamlit app
st.title("Loan Eligibility Prediction")

st.write("Enter the details of the applicant to predict loan eligibility.")

# Dynamically generate input fields based on the dataset features
user_input = {}
for col in X.columns:
    value = st.number_input(f"{col}", value=float(X[col].mean()))
    user_input[col] = value

# Predict when the button is clicked
if st.button("Predict Loan Eligibility"):
    input_df = pd.DataFrame([user_input])
    logging.info("Making prediction...")
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The applicant is eligible for the loan!")
    else:
        st.error("The applicant is not eligible for the loan.")
    
    # Evaluate the model
    logging.info("Evaluating the model...")
    acc, report, matrix = model_lr.evaluate_model(model, X_test, y_test)
    st.write("### Model Evaluation")
    st.write(f"Accuracy: {acc:.2f}")
    st.write("#### Classification Report")
    st.text(report)
    st.write("#### Confusion Matrix")
    st.write(matrix)
