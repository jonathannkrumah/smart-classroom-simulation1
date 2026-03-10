# Extract thresholds from model"""
Extract environmental thresholds from trained Random Forest model
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

def extract_tree_thresholds(model, feature_names):
    """
    Extract decision thresholds from all trees in Random Forest
    
    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
    
    Returns:
        dict: Thresholds for each feature
    """
    thresholds = {feature: [] for feature in feature_names}
    
    # Extract thresholds from each tree
    for tree in model.estimators_:
        tree_ = tree.tree_
        
        # Get feature indices and thresholds for split nodes
        feature_indices = tree_.feature
        threshold_values = tree_.threshold
        
        # Collect thresholds for each feature
        for i, feature_idx in enumerate(feature_indices):
            if feature_idx != -2:  # -2 indicates leaf node
                feature_name = feature_names[feature_idx]
                thresholds[feature_name].append(threshold_values[i])
    
    # Calculate statistics for each feature
    threshold_stats = {}
    for feature, values in thresholds.items():
        if values:
            threshold_stats[feature] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'count': len(values)
            }
    
    return threshold_stats

def extract_feature_importance(model, feature_names):
    """
    Extract feature importance from Random Forest
    
    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
    
    Returns:
        DataFrame: Feature importance scores
    """
    importance = model.feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df_importance

def get_optimal_thresholds(model_path, feature_names):
    """
    Main function to extract optimal thresholds from trained model
    
    Args:
        model_path: Path to trained model file
        feature_names: List of feature names
    
    Returns:
        dict: Optimal thresholds and recommendations
    """
    print("Extracting thresholds from trained model...")
    print("-" * 50)
    
    # Load model
    model = joblib.load(model_path)
    
    # Extract thresholds
    threshold_stats = extract_tree_thresholds(model, feature_names)
    
    # Extract feature importance
    importance_df = extract_feature_importance(model, feature_names)
    
    # Determine optimal thresholds
    optimal_thresholds = {}
    
    for feature, stats in threshold_stats.items():
        # Use median as optimal threshold
        # This represents the point where model splits most frequently
        optimal = stats['median']
        
        # Add context-specific adjustments based on known standards
        if 'co2' in feature.lower():
            # ASHRAE standard: 1000ppm
            recommended = min(optimal, 1000)
            optimal_thresholds[feature] = {
                'warning': optimal,
                'critical': min(optimal * 1.2, 1000),
                'description': f"CO₂ above {optimal:.0f}ppm indicates reduced focus"
            }
        elif 'temperature' in feature.lower():
            optimal_thresholds[feature] = {
                'optimal_range': [20, 26],
                'warning_low': 19,
                'warning_high': 27,
                'critical_low': 18,
                'critical_high': 28,
                'description': "Temperature outside 20-26°C affects comfort"
            }
        elif 'light' in feature.lower():
            optimal_thresholds[feature] = {
                'minimum': 300,
                'optimal': 500,
                'maximum': 800,
                'description': "Light below 300lux or above 800lux causes eye strain"
            }
        elif 'noise' in feature.lower():
            optimal_thresholds[feature] = {
                'warning': 55,
                'critical': 65,
                'description': "Noise above 55dB distracts attention"
            }
        elif 'humidity' in feature.lower():
            optimal_thresholds[feature] = {
                'optimal_range': [30, 60],
                'warning_low': 25,
                'warning_high': 65,
                'description': "Humidity outside 30-60% affects comfort"
            }
    
    # Print summary
    print("\n📊 FEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))
    
    print("\n🎯 EXTRACTED THRESHOLDS")
    for feature, thresholds in optimal_thresholds.items():
        print(f"\n{feature.upper()}:")
        for key, value in thresholds.items():
            if key != 'description':
                print(f"  • {key}: {value}")
        print(f"  • {thresholds['description']}")
    
    return optimal_thresholds, importance_df

def generate_control_rules(thresholds):
    """
    Generate control logic rules from extracted thresholds
    
    Args:
        thresholds: Thresholds dictionary from get_optimal_thresholds
    
    Returns:
        dict: Control rules for simulation
    """
    rules = {}
    
    for feature, config in thresholds.items():
        if 'co2' in feature.lower():
            rules['co2_control'] = {
                'warning': config['warning'],
                '