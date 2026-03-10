import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Generate dummy training data
np.random.seed(42)
n_samples = 1000

# Create synthetic data based on your thresholds
temperature = np.random.uniform(18, 32, n_samples)
co2 = np.random.uniform(400, 1200, n_samples)
humidity = np.random.uniform(20, 80, n_samples)
light = np.random.uniform(100, 900, n_samples)
noise = np.random.uniform(35, 75, n_samples)

# Create target based on realistic rules (for dummy model)
# This is just for testing - your real model will use actual data
target = []
for i in range(n_samples):
    # Non-conducive if any condition is poor
    if (co2[i] > 800 or 
        temperature[i] > 27 or temperature[i] < 18 or
        light[i] < 200 or light[i] > 800 or
        noise[i] > 65 or
        humidity[i] > 70 or humidity[i] < 30):
        target.append(1)  # non-conducive
    else:
        target.append(0)  # conducive

# Create DataFrame
X = pd.DataFrame({
    'temperature': temperature,
    'co2': co2,
    'humidity': humidity,
    'light': light,
    'noise': noise
})
y = np.array(target)

# Train a simple Random Forest
print("Training dummy Random Forest model...")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X, y)

# Save the model
model_path = 'models/random_forest_model.pkl'
joblib.dump(model, model_path)
print(f"✅ Dummy model saved to {model_path}")

# Test the model
test_features = pd.DataFrame([{
    'temperature': 22,
    'co2': 450,
    'humidity': 50,
    'light': 400,
    'noise': 45
}])
pred = model.predict(test_features)[0]
proba = model.predict_proba(test_features)[0]
print(f"Test prediction: {'non-conducive' if pred == 1 else 'conducive'} (confidence: {max(proba):.2f})")