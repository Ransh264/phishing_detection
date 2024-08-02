import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def select_features(df):
    # 'Result' column is not included in X
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    # Encode categorical features
    if X.select_dtypes(include=['object']).shape[1] > 0:
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
    
    # Feature selection using Mutual Information
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_new = selector.fit_transform(X, y)
    
    return X_new, selector

# old_df has already been loaded and preprocessed
if __name__ == "__main__":
    # Import the preprocessed data
    from preprocessing import old_df
    
    X_new, selector = select_features(old_df)
    
    # Print feature scores
    feature_scores = selector.scores_
    print("Feature scores:")
    for i, score in enumerate(feature_scores):
        print(f"Feature {i}: Score = {score}")
