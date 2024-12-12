from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model(X_train, y_train, model_type='logistic'):
    """Train a model."""
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        model = SVC(random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'logistic', 'random_forest', or 'svm'.")

    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save the trained model to a file."""
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    from feature_engineering import split_data, balance_data
    import pandas as pd

    processed_path = "data/processed/processed_credit_data.csv"
    model_path = "models/credit_risk_model.pkl"

    # Load data
    df = pd.read_csv(processed_path)

    # Prepare and balance data
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    # Train and save the model
    model = train_model(X_train_balanced, y_train_balanced, model_type='logistic')
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

