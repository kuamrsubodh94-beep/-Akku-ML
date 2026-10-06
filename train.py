import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Create a dummy synthetic dataset for demonstration purposes
print("Creating synthetic dataset...")
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(20, 80, size=n_samples),
    'bmi': np.random.uniform(15.0, 40.0, size=n_samples),
    'genetic_risk': np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2]),
    'smoking': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
    'salt_intake': np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.5, 0.3]),
}

df = pd.DataFrame(data)

# Define a deterministic logic with some noise to generate the target 'hypertension'
# Risk score calculation
risk_score = (
    (df['age'] * 0.05) + 
    ((df['bmi'] - 22) * 0.1) + 
    (df['genetic_risk'] * 0.8) + 
    (df['smoking'] * 0.5) + 
    (df['salt_intake'] * 0.6)
)
# If risk score > threshold, highly likely to have hypertension
noise = np.random.normal(0, 0.5, size=n_samples)
df['hypertension'] = ((risk_score + noise) > 3.0).astype(int)

# Save the dataset to a CSV file
df.to_csv('hypertension_data.csv', index=False)
print("Dataset saved as 'hypertension_data.csv'")

# 2. Split Data into Features (X) and Target (y)
X = df.drop(columns=['hypertension'])
y = df['hypertension']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model and scaler
with open('hypertension_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler successfully saved!")
