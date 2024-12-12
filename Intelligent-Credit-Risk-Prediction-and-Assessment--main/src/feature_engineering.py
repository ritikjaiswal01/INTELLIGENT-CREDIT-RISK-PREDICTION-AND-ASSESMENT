import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def split_data(df, target_column='Has Default'):
    """Split the dataset into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def balance_data(X_train, y_train):
    """Apply SMOTE to balance the classes."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

if __name__ == "__main__":
    processed_path = "data/processed/processed_credit_data.csv"

    # Load processed data
    df = pd.read_csv(processed_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Balance data
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    print("Feature engineering completed.")

