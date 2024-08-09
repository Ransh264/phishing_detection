import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from load_data import load_arff
import numpy as np

def plot_confusion_matrix(y_true, y_pred):
    # Convert labels to integers if they are not already
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])  # Ensure labels are correct
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Phishing', 'Phishing'])
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

    # tick labels
    num_labels = len(disp.display_labels)
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    ax.set_xticklabels(disp.display_labels)
    ax.set_yticklabels(disp.display_labels)

    plt.savefig('confusion_matrix.png')
    plt.show()
    print("Confusion matrix plot saved.")

# Load the model
model = joblib.load('saved_models/random_forest_model.pkl')

# Loading dataset 
test_df = load_arff(r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff') 
test_df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Prepare data
X_test = test_df.drop('Result', axis=1)
y_test = test_df['Result']

# Make predictions
y_pred = model.predict(X_test)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)
