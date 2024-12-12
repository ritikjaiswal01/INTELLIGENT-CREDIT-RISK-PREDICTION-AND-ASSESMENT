import pandas as pd
import numpy as np
import os

# Define the number of samples
sample_size = 40000

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with feature-target relationships
data = {
    'Age': np.random.choice(range(18, 70), sample_size, replace=True),  # Allow repetition
    'Annual Income': np.random.choice(range(20000, 150000, 500), sample_size, replace=True),
    'Credit Score': np.random.choice(range(300, 850), sample_size, replace=True),
    'Loan Amount': np.round(np.random.uniform(5000, 50000, sample_size), 2),
    'Existing Loan': np.random.choice([0, 1], sample_size, replace=True),
    'Debt to Income': np.round(np.random.uniform(0.1, 1.0, sample_size), 2),
}

# Generate target variable 'Has Default' with a relationship
def calculate_default(row):
    score = 0
    if row['Credit Score'] < 600:
        score += 3
    if row['Debt to Income'] > 0.5:
        score += 2
    if row['Existing Loan'] == 1:
        score += 1
    if row['Annual Income'] < 50000:
        score += 1
    return 1 if score >= 4 else 0

df = pd.DataFrame(data)
df['Has Default'] = df.apply(calculate_default, axis=1)

# Save the dataset
folder_path = "data/raw"
os.makedirs(folder_path, exist_ok=True)

file_path = os.path.join(folder_path, "raw_credit_data.csv")
df.to_csv(file_path, index=False)

# Print confirmation and class distribution
print(f"Dataset with {sample_size} samples generated and saved to {file_path}")
print("\nClass distribution in the generated dataset:")
print(df['Has Default'].value_counts())




