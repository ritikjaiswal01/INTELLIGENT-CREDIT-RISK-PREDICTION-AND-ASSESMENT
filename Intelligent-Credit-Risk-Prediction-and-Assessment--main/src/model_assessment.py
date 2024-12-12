from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def load_model(filepath):
    """Load a saved model."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    from feature_engineering import split_data
    import pandas as pd

    processed_path = "data/processed/processed_credit_data.csv"
    model_path = "models/credit_risk_model.pkl"

    # Load data and split
    df = pd.read_csv(processed_path)
    _, X_test, _, y_test = split_data(df)

    # Load model
    model = load_model(model_path)

    # Evaluate model
    evaluate_model(model, X_test, y_test)
