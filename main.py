import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import logging
from utils.preprocessing import load_data, preprocess_data
from models import model_lr
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Optional: Set working directory to the script's location
os.chdir(os.path.abspath(os.path.dirname(__file__)))

# Load and preprocess data
logging.info("Loading and preprocessing data...")
df = load_data("data/credit.csv")
X, y = preprocess_data(df)

# Split data into training and test sets
logging.info("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model using modular function
logging.info("Training logistic regression model...")
model = model_lr.train_model(X_train, y_train)

# Evaluate the model using modular function
logging.info("Evaluating the model...")
acc, report, matrix = model_lr.evaluate_model(model, X_test, y_test)

# Print evaluation results
print("\nConfusion Matrix:")
print(matrix)
print("\nClassification Report:")
print(report)
print(f"Accuracy Score: {acc:.2f}")
