import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# path to find load_data
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data import load_arff

# Load the data
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

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

#ensures that saved_models exists
os.makedirs('saved_models', exist_ok=True)

# Save the model
joblib.dump(model, 'saved_models/random_forest_model.pkl')
print("Model saved as 'saved_models/random_forest_model.pkl'.")
print("Model trained successfully:", model)
