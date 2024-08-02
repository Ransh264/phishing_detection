# Phishing Website Detection Using Machine Learning

## Project Overview
This project aims to detect phishing websites using machine learning techniques. The detection model is trained on a dataset of phishing and legitimate websites, with the goal of classifying websites based on various features.

## Project Structure
The project is organized into the following folders and files:

`saved_models/`: Contains saved model files.
    - `random_forest_model.pkl`: The trained Random Forest model.
- `src/`: Contains source code files.
  - `model_training.py`: Script to train the machine learning model.
  - `model_evaluation.py`: Script to evaluate the trained model.
  - `visualizations.py`: Script to generate visualizations such as feature importance.
  - `visualizations1.py`: Script to generate confusion matrix visualizations.
  - `load_data.py`: Script to load and preprocess the dataset.

- `README.md`: This file, which provides an overview and instructions for the project.

## Dataset
The dataset used for this project includes various features related to website URLs. It is formatted in ARFF (Attribute-Relation File Format) and includes the following attributes:
- `having_IP_Address`
- `URL_Length`
- `Shortining_Service`
- `having_At_Symbol`
- `double_slash_redirecting`
- `Prefix_Suffix`
- `having_Sub_Domain`
- `SSLfinal_State`
- `Domain_registeration_length`
- `Favicon`
- `port`
- `HTTPS_token`
- `Request_URL`
- `URL_of_Anchor`
- `Links_in_tags`
- `SFH`
- `Submitting_to_email`
- `Abnormal_URL`
- `Redirect`
- `on_mouseover`
- `RightClick`

## Installation and Setup
   cd phishing_detection

## Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install the required packages:
pip install -r requirements.txt