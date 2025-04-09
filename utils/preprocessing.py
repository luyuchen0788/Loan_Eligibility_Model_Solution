import pandas as pd

def load_data(file_path):
    """Load the loan dataset from a CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess the dataset for model training"""
    # Drop non-predictive identifier
    df = df.drop(columns=["Loan_ID"])

    # Handle missing values
    df["Gender"].fillna("Male", inplace=True)
    df["Married"].fillna("Yes", inplace=True)
    df["Dependents"].fillna("0", inplace=True)
    df["Self_Employed"].fillna("No", inplace=True)
    df["Credit_History"].fillna(1.0, inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)

    # Encode target variable
    df["Loan_Approved"] = df["Loan_Approved"].map({"Y": 1, "N": 0})

    # One-hot encode categorical variables
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save cleaned data before splitting
    df.to_csv("data/loan_eligibility.csv", index=False)

    # Split into features and target
    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"]

    return X, y
