import sys
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Add the path to find load_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data import load_arff

# Load the dataset
df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff')

# Verify class distribution
class_distribution = df['Result'].value_counts()
print("Class Distribution:\n", class_distribution)

# Split data into features and target
X = df.drop('Result', axis=1)
y = df['Result']

# Apply SMOTE for oversampling (if phishing is the minority class)
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify new class distribution
new_class_distribution = pd.Series(y_resampled).value_counts()
print("New Class Distribution:\n", new_class_distribution)
