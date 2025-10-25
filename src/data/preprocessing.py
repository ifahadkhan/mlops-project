import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def normalize_features(df, columns):
    """Normalize specified columns using StandardScaler."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

def save_processed_data(df, filepath):
    """Save processed data to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    data = load_data('data/raw/dataset.csv')
    processed_data, scaler = normalize_features(data, ['feature1', 'feature2'])
    save_processed_data(processed_data, 'data/processed/normalized_data.csv')
