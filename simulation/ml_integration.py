# ML model loader & predictor
import joblib
import numpy as np
import pandas as pd

# Load your trained model
MODEL_PATH = "models/random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_environment(features):
    """
    Predicts if environment is conducive or non-conducive
    Returns: (prediction, confidence)
    """
    # Convert features to DataFrame
    df = pd.DataFrame([features])
    
    # Ensure correct column order (same as training)
    columns = ['temperature', 'co2', 'humidity', 'light', 'noise']
    df = df[columns]
    
    # Get prediction and probability
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    
    # Map prediction to labels
    labels = {0: "conducive", 1: "non-conducive"}
    confidence = max(proba)
    
    return labels[prediction], confidence

def extract_thresholds():
    """Extracts decision thresholds from Random Forest model"""
    # Implementation depends on your model structure
    # This would analyze tree split points
    pass