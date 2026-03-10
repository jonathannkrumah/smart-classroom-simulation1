# Data cleaning & fusion"""
Data preprocessing utilities for IoT classroom datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(iot_path, thermal_path=None):
    """
    Load and preprocess IoT classroom data
    
    Args:
        iot_path: Path to IoT sensor data CSV
        thermal_path: Path to thermal comfort data (optional)
    
    Returns:
        DataFrame: Preprocessed dataset ready for ML
    """
    print("Loading IoT dataset...")
    df_iot = pd.read_csv(iot_path)
    
    # Basic preprocessing
    print("Preprocessing data...")
    
    # Handle datetime
    if 'timestamp' in df_iot.columns:
        df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'])
        df_iot.set_index('timestamp', inplace=True)
    
    # Handle missing values
    numeric_cols = df_iot.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_iot[col].fillna(df_iot[col].median(), inplace=True)
    
    # Handle outliers (cap at 99th percentile)
    for col in numeric_cols:
        upper = df_iot[col].quantile(0.99)
        lower = df_iot[col].quantile(0.01)
        df_iot[col] = df_iot[col].clip(lower, upper)
    
    # If thermal data provided, merge
    if thermal_path and os.path.exists(thermal_path):
        print("Loading thermal comfort data...")
        df_thermal = pd.read_csv(thermal_path)
        # Merge logic would go here
        # For now, just return IoT data
    
    print(f"Dataset shape: {df_iot.shape}")
    print(f"Features: {list(df_iot.columns)}")
    
    return df_iot

def create_target_variable(df, co2_threshold=800, temp_range=(20, 26)):
    """
    Create target variable for supervised learning
    
    Args:
        df: DataFrame with environmental variables
        co2_threshold: CO2 ppm threshold for 'non-conducive'
        temp_range: Temperature range for 'conducive'
    
    Returns:
        DataFrame with added 'attention_state' column
    """
    df = df.copy()
    
    # Rule-based target creation (for demonstration)
    # In practice, this should come from actual student focus labels
    
    conditions = []
    
    # CO2 condition
    if 'co2' in df.columns:
        conditions.append(df['co2'] > co2_threshold)
    
    # Temperature condition
    if 'temperature' in df.columns:
        temp_condition = (df['temperature'] < temp_range[0]) | (df['temperature'] > temp_range[1])
        conditions.append(temp_condition)
    
    # Humidity condition
    if 'humidity' in df.columns:
        conditions.append((df['humidity'] < 30) | (df['humidity'] > 70))
    
    # Light condition
    if 'light' in df.columns:
        conditions.append(df['light'] < 200)
    
    # Combine conditions
    if conditions:
        df['attention_state'] = np.logical_or.reduce(conditions).astype(int)
    else:
        df['attention_state'] = 0  # Default conducive
    
    return df

def prepare_for_training(df, target_col='attention_state', test_size=0.2, random_state=42):
    """
    Prepare data for machine learning training
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Proportion for test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col 
                    and col not in ['timestamp', 'date', 'time']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Example usage
    print("Data Preprocessor Module")
    print("-" * 40)
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(23, 3, n_samples),
        'co2': np.random.normal(600, 150, n_samples),
        'humidity': np.random.normal(50, 10, n_samples),
        'light': np.random.normal(400, 100, n_samples),
        'noise': np.random.normal(50, 8, n_samples)
    })
    
    # Add target
    sample_data = create_target_variable(sample_data)
    
    # Prepare for training
    X_train, X_test, y_train, y_test, scaler = prepare_for_training(sample_data)
    
    print("\n✅ Data preprocessing complete")