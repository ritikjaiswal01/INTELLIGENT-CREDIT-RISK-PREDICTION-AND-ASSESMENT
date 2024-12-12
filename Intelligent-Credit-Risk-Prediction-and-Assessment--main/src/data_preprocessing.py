import pandas as pd
import os

def load_data(filepath):
    """Load data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the dataset by handling missing and invalid data."""
    # Drop rows with missing values
    df = df.dropna()

    # Ensure required columns exist
    required_columns = ['Age', 'Annual Income', 'Credit Score', 'Loan Amount', 'Has Default']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df

def save_data(df, filepath):
    """Save the processed data to a CSV file."""
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    raw_path = "data/raw/raw_credit_data.csv"
    processed_path = "data/processed/processed_credit_data.csv"

    # Load and clean data
    data = load_data(raw_path)
    cleaned_data = clean_data(data)

    # Save processed data
    save_data(cleaned_data, processed_path)
    print(f"Processed data saved to {processed_path}")
