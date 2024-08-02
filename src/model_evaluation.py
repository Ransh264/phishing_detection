import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# path to find load_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data import load_arff

# Load the saved model
model = joblib.load('saved_models/random_forest_model.pkl')

# Load and preprocess the data
training_df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff')

def preprocess_data(df):
    df.fillna(method='ffill', inplace=True)
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df

training_df = preprocess_data(training_df)

# Split the data into features and target
X_test = training_df.drop('Result', axis=1)
y_test = training_df['Result']

# Standardize features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)  # Ensure to use the same scaler as used in training

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

