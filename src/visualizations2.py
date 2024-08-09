import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='o', color='orange', label=f'ROC curve (AUC = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.show()

# Plot ROC Curve
plot_roc_curve(y_test, y_scores)
