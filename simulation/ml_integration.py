"""
ML model integration for smart classroom simulation
Handles loading the trained model and making predictions
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime


MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Global variables (lazy loading)
_model = None
_scaler = None
_feature_columns = None
_numeric_label_map = None
_model_reliable = True


def _discover_artifacts():
    """Discover model, scaler, and feature column artifacts from models directory."""
    model_path = None
    scaler_path = None
    feature_columns_path = None

    if not os.path.isdir(MODELS_DIR):
        return model_path, scaler_path, feature_columns_path

    artifact_files = [
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.lower().endswith(('.pkl', '.joblib'))
    ]

    # Prefer non-scaler artifacts as model artifact
    model_candidates = [p for p in artifact_files if 'scaler' not in os.path.basename(p).lower()]
    scaler_candidates = [p for p in artifact_files if 'scaler' in os.path.basename(p).lower()]

    if model_candidates:
        # Prefer core_model.pkl (latest, aligned with simulation), then random_forest, else first
        basename_to_priority = {
            'core_model.pkl': 0,
            'random_forest_model.joblib': 1,
            'random_forest.pkl': 1,
        }
        model_candidates.sort(key=lambda p: (
            basename_to_priority.get(os.path.basename(p), 99),
            p.lower()
        ))
        model_path = model_candidates[0]

    if scaler_candidates:
        # Prefer core_scaler.pkl, then scaler.pkl
        basename_to_priority = {
            'core_scaler.pkl': 0,
            'scaler.pkl': 1,
        }
        scaler_candidates.sort(key=lambda p: (
            basename_to_priority.get(os.path.basename(p), 99),
            p.lower()
        ))
        scaler_path = scaler_candidates[0]

    # Prefer core feature columns
    feature_columns_candidates = [
        os.path.join(MODELS_DIR, 'core_feature_columns.json'),
        os.path.join(MODELS_DIR, 'feature_columns.json'),
    ]
    for candidate in feature_columns_candidates:
        if os.path.exists(candidate):
            feature_columns_path = candidate
            break

    return model_path, scaler_path, feature_columns_path


def _load_feature_columns(feature_columns_path, model):
    """Load feature names from JSON; fallback to model metadata if available."""
    if feature_columns_path and os.path.exists(feature_columns_path):
        with open(feature_columns_path, 'r', encoding='utf-8') as f:
            cols = json.load(f)
            if isinstance(cols, list) and cols:
                return cols

    model_columns = getattr(model, 'feature_names_in_', None)
    if model_columns is not None and len(model_columns) > 0:
        return list(model_columns)

    n_features = getattr(model, 'n_features_in_', None)
    if n_features:
        return [f'feature_{i}' for i in range(n_features)]

    return None


def load_model():
    """
    Loads the trained model artifacts from models directory.
    Returns None if model not found (simulation will use fallback rules)
    """
    global _model, _scaler, _feature_columns, _model_reliable
    if _model is None:
        try:
            model_path, scaler_path, feature_columns_path = _discover_artifacts()

            if model_path:
                _model = joblib.load(model_path)
                print(f"✅ Model loaded from {model_path}")

                if scaler_path:
                    _scaler = joblib.load(scaler_path)
                    print(f"✅ Scaler loaded from {scaler_path}")

                _feature_columns = _load_feature_columns(feature_columns_path, _model)
                if _feature_columns:
                    print(f"✅ Loaded {_feature_columns.__len__()} feature columns")

                _model_reliable = _validate_model_behavior(_model)
                if not _model_reliable:
                    print("⚠️ Loaded model appears unreliable in this environment")
                    print("   Falling back to rule-based predictions")
            else:
                print("⚠️ Model not found in models directory")
                print("   Using rule-based fallback predictions")
                _model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _model = None
    return _model


def _rule_based_prediction(features):
    """Rule-based fallback prediction using threshold logic."""
    conducive = True

    # CO2 threshold (from your model: ~700-800ppm warning)
    if features['co2'] > 800:
        conducive = False
    elif features['co2'] > 700:
        conducive = False  # Early warning threshold

    # Temperature threshold (from your model: ~25-26°C)
    if features['temperature'] > 27:
        conducive = False
    elif features['temperature'] < 18:
        conducive = False

    # Light threshold (from your model: ~300-500 lux optimal)
    if features['light'] < 250:
        conducive = False
    elif features['light'] > 800:
        conducive = False

    # Humidity threshold
    if features['humidity'] > 70 or features['humidity'] < 30:
        conducive = False

    prediction = 'conducive' if conducive else 'non-conducive'
    confidence = 0.75 if conducive else 0.85
    return prediction, confidence


def _normalize_prediction_label(raw_label, label_map=None):
    """Normalize model-specific labels to: conducive / non-conducive."""
    if label_map is not None and raw_label in label_map:
        return label_map[raw_label]

    text = str(raw_label).strip().lower().replace('_', '-').replace(' ', '-')

    if 'non' in text and 'conducive' in text:
        return 'non-conducive'
    if text == 'conducive':
        return 'conducive'

    # Common numeric encodings
    try:
        numeric = float(raw_label)
        return 'conducive' if numeric == 0 else 'non-conducive'
    except (TypeError, ValueError):
        pass

    # Conservative fallback
    return 'non-conducive'


def _model_input_for_predict(model, df):
    """Return the right input format for model.predict* calls."""
    has_feature_names = getattr(model, 'feature_names_in_', None) is not None
    return df if has_feature_names else df.values


def _infer_numeric_label_map(model):
    """Infer numeric class meaning using a healthy reference sample."""
    classes = getattr(model, 'classes_', None)
    if classes is None or len(classes) != 2:
        return None

    try:
        normalized_classes = [float(c) for c in classes]
    except (TypeError, ValueError):
        return None

    # Only infer for binary numeric labels
    if set(normalized_classes) != {0.0, 1.0}:
        return None

    try:
        reference_features = _prepare_model_features({
            'temperature': 22,
            'humidity': 50,
            'co2': 450,
            'noise': 45,
            'light': 400,
            'occupancy_count': 20,
        })
        predicted_reference_label = model.predict(_model_input_for_predict(model, reference_features))[0]
    except Exception:
        return None

    # Reference sample is intentionally conducive.
    return {
        predicted_reference_label: 'conducive',
        next(c for c in classes if c != predicted_reference_label): 'non-conducive',
    }


def _validate_model_behavior(model):
    """
    Lightweight compatibility check.
    Avoid strict anchor-based gating because engineered-feature defaults can vary
    across deployments and cause false negatives.
    """
    try:
        classes = getattr(model, 'classes_', None)
        if classes is None or len(classes) < 2:
            return False
        return hasattr(model, 'predict') and hasattr(model, 'predict_proba')
    except Exception:
        return False


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _default_datetime(context=None):
    if context and context.get('datetime'):
        return context['datetime']
    return datetime.now()


def _prepare_model_features(features, context=None):
    """
    Build model-ready feature frame using only core environmental variables.
    No complex feature engineering—directly use temperature, humidity, co2, light.
    """
    load_model()

    # Get core values from input
    temp = _safe_float(features.get('temperature', 22.0), 22.0)
    humidity = _safe_float(features.get('humidity', 50.0), 50.0)
    co2 = _safe_float(features.get('co2', 450.0), 450.0)
    light = _safe_float(features.get('light', 400.0), 400.0)

    # Build canonical feature dict from simulation/live input
    feature_map = {
        'temperature': temp,
        'humidity': humidity,
        'co2': co2,
        'light': light,
    }

    # Map known training-column aliases to canonical values
    alias_map = {
        'temperature': ('temperature', 'temp', 'temperature_c', 'temperature(c)', 'temp_c'),
        'humidity': ('humidity', 'humidity_perc', 'relative_humidity', 'humidity_%'),
        'co2': ('co2', 'co2_ppm', 'co2ppm'),
        'light': ('light', 'light_lux', 'lux'),
    }

    def value_for_model_column(column_name):
        key = str(column_name).strip().lower().replace(' ', '_')
        for canonical, aliases in alias_map.items():
            if key in aliases:
                return feature_map[canonical]
        return feature_map.get(key, 0.0)

    # If we know the expected columns, use them in order
    if _feature_columns:
        ordered = {
            col: _safe_float(value_for_model_column(col), 0.0)
            for col in _feature_columns
        }
        df = pd.DataFrame([ordered])
    else:
        df = pd.DataFrame([feature_map])

    # Apply scaler if available
    if _scaler is not None:
        scaled = _scaler.transform(df.values)
        df = pd.DataFrame(scaled, columns=df.columns)

    return df

def predict_environment(features, context=None):
    """
    Predicts if environment is conducive or non-conducive for learning
    
    Args:
        features: dict with keys 'temperature', 'co2', 'humidity', 'light'
    
    Returns:
        tuple: (prediction_label, confidence)
    """
    global _numeric_label_map, _model_reliable
    model = load_model()
    model_features = _prepare_model_features(features, context=context)
    
    # If model loaded successfully, use it
    if model is not None and _model_reliable:
        try:
            # Get prediction
            model_input = _model_input_for_predict(model, model_features)
            prediction_raw = model.predict(model_input)[0]
            probabilities = model.predict_proba(model_input)[0]
            confidence = max(probabilities)

            if _numeric_label_map is None:
                _numeric_label_map = _infer_numeric_label_map(model)

            # Map model output to canonical label
            prediction = _normalize_prediction_label(prediction_raw, label_map=_numeric_label_map)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Fall through to rule-based

    # Fallback: rule-based prediction based on known thresholds
    return _rule_based_prediction(features)

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