"""
ML model integration for smart classroom simulation
Handles loading the trained model and making predictions
"""

import numpy as np
import pandas as pd
import joblib
import os

# Path to your trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl')

# Global variable for model (lazy loading)
_model = None

def load_model():
    """
    Loads the trained Random Forest model
    Returns None if model not found (simulation will use fallback rules)
    """
    global _model
    if _model is None:
        try:
            if os.path.exists(MODEL_PATH):
                _model = joblib.load(MODEL_PATH)
                print(f"✅ Model loaded from {MODEL_PATH}")
            else:
                print(f"⚠️ Model not found at {MODEL_PATH}")
                print("   Using rule-based fallback predictions")
                _model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _model = None
    return _model

def predict_environment(features):
    """
    Predicts if environment is conducive or non-conducive for learning
    
    Args:
        features: dict with keys 'temperature', 'co2', 'humidity', 'light', 'noise'
    
    Returns:
        tuple: (prediction_label, confidence)
    """
    model = load_model()
    
    # If model loaded successfully, use it
    if model is not None:
        try:
            # Convert features to DataFrame with correct column order
            df = pd.DataFrame([features])
            columns = ['temperature', 'co2', 'humidity', 'light', 'noise']
            df = df[columns]
            
            # Get prediction
            prediction_num = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            confidence = max(probabilities)
            
            # Map numeric prediction to label
            # Adjust based on your model's label encoding
            if hasattr(model, 'classes_'):
                if model.classes_[0] == 'conducive' or model.classes_[0] == 0:
                    prediction = 'conducive' if prediction_num == 0 else 'non-conducive'
                else:
                    prediction = 'non-conducive' if prediction_num == 0 else 'conducive'
            else:
                prediction = 'conducive' if prediction_num == 0 else 'non-conducive'
            
            return prediction, confidence
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Fall through to rule-based
    
    # Fallback: rule-based prediction based on known thresholds
    # These thresholds come from your research findings
    conducive = True
    reasons = []
    
    # CO2 threshold (from your model: ~700-800ppm warning)
    if features['co2'] > 800:
        conducive = False
        reasons.append(f"CO2 high ({features['co2']:.0f}ppm)")
    elif features['co2'] > 700:
        conducive = False  # Early warning threshold
        reasons.append(f"CO2 elevated ({features['co2']:.0f}ppm)")
    
    # Temperature threshold (from your model: ~25-26°C)
    if features['temperature'] > 27:
        conducive = False
        reasons.append(f"Temperature high ({features['temperature']:.1f}°C)")
    elif features['temperature'] < 18:
        conducive = False
        reasons.append(f"Temperature low ({features['temperature']:.1f}°C)")
    
    # Light threshold (from your model: ~300-500 lux optimal)
    if features['light'] < 250:
        conducive = False
        reasons.append(f"Light too dim ({features['light']:.0f}lux)")
    elif features['light'] > 800:
        conducive = False
        reasons.append(f"Light too bright ({features['light']:.0f}lux)")
    
    # Noise threshold
    if features['noise'] > 65:
        conducive = False
        reasons.append(f"Noise high ({features['noise']:.0f}dB)")
    
    # Humidity threshold
    if features['humidity'] > 70 or features['humidity'] < 30:
        conducive = False
        reasons.append(f"Humidity suboptimal ({features['humidity']:.0f}%)")
    
    prediction = 'conducive' if conducive else 'non-conducive'
    confidence = 0.75 if conducive else 0.85
    
    return prediction, confidence

def test_model_prediction():
    """
    Test function to verify ML predictions
    """
    print("\n🔍 Testing ML Model Predictions")
    print("-" * 40)
    
    test_cases = [
        {'temperature': 22, 'co2': 450, 'humidity': 50, 'light': 400, 'noise': 45},  # Ideal
        {'temperature': 28, 'co2': 900, 'humidity': 65, 'light': 200, 'noise': 70},  # Poor
        {'temperature': 25, 'co2': 750, 'humidity': 55, 'light': 300, 'noise': 55},  # Borderline
    ]
    
    for i, case in enumerate(test_cases, 1):
        pred, conf = predict_environment(case)
        print(f"\nTest Case {i}:")
        print(f"  Conditions: T={case['temperature']}°C, CO2={case['co2']}ppm, "
              f"Light={case['light']}lux")
        print(f"  Prediction: {pred.upper()} (confidence: {conf:.1%})")

if __name__ == "__main__":
    test_model_prediction()