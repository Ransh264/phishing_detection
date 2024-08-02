import matplotlib.pyplot as plt
import joblib
from load_data import load_arff

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, len(features)])
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    print("Feature importance plot saved.")

# Load the model
model = joblib.load('saved_models/random_forest_model.pkl')

# Loaded dataset 
test_df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff')  # Update this path to your actual test dataset
test_df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Prepared data
X_test = test_df.drop('Result', axis=1)

# Plot feature importance
plot_feature_importance(model, X_test.columns)
