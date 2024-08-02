import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
from load_data import load_arff

# the file paths
old_df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\.old.arff')
training_df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff')

# preprocessing steps
def preprocess_data(df):
    # Handle missing values (if any)
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values

    # Convert categorical columns to numeric
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df

# Preprocess the data
old_df = preprocess_data(old_df)
training_df = preprocess_data(training_df)

# Split the data into features and target
X = training_df.drop('Result', axis=1)
y = training_df['Result']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data preprocessing complete.")
