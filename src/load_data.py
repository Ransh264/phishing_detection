import pandas as pd
from scipy.io import arff
from docx import Document

# Function to load ARFF files
def load_arff(filename):
    data, meta = arff.loadarff(filename)
    df = pd.DataFrame(data)
    # Convert byte columns to string
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode('utf-8')
    return df

# Load the datasets
old_arff_file = r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\.old.arff'
training_arff_file = r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Training Dataset.arff'
docx_file = r'C:\Users\prerna singh\OneDrive\Desktop\phishing detection\Phishing Websites Features.docx'

old_df = load_arff(old_arff_file)
training_df = load_arff(training_arff_file)

# Read the DOCX file
doc = Document(docx_file)
doc_text = '\n'.join([para.text for para in doc.paragraphs])

# Print the first few rows of each DataFrame
print("Old ARFF File Data:")
print(old_df.head())
print("\nTraining ARFF File Data:")
print(training_df.head())

# Print DOCX file content
print("\nDOCX File Content:")
print(doc_text)
