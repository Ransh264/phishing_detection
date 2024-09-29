import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from load_data import load_arff

# Load the data
training_df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff')

# Preprocessing steps
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

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit Grid Search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Best Parameters: {accuracy}")

report = classification_report(y_test, y_pred)
print("Classification Report with Best Parameters:")
print(report)

# Save the best model
os.makedirs('saved_models', exist_ok=True)
joblib.dump(best_model, 'saved_models/random_forest_best_model.pkl')
print("Best model saved as 'saved_models/random_forest_best_model.pkl'.")
print("Model tuned and trained successfully:", best_model)
