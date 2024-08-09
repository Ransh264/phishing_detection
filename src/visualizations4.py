import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.io import arff

# Load dataset
def load_data(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df

# Path to dataset
file_path = 'C:\\Users\\prerna singh\\OneDrive\\Desktop\\phishing detection\\Training Dataset.arff'

# Load the data
df = load_data(file_path)

# Encode categorical features and target variable
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Separate features and target variable
X = df.drop('Result', axis=1)
y = df['Result']

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Set up K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Print the results
print(f'K-Fold Cross-Validation Accuracy Scores: {cv_results}')
print(f'Mean Accuracy: {cv_results.mean():.2f}')
print(f'Standard Deviation: {cv_results.std():.2f}')

# Plot K-Fold Cross-Validation Results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_results) + 1), cv_results, marker='o', linestyle='-', color='blue')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('K-Fold Cross-Validation Results')
plt.grid(True)
plt.savefig('k_fold_cross_validation_results.png')  # Save the plot as a .png file
plt.show()  # Display the plot
