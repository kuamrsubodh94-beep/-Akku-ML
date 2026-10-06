# -Akku-ML
# Hypertension Prediction ML Project

This repository contains a simple, end-to-end Machine Learning pipeline to predict the likelihood of **Hypertension** based on basic demographic and lifestyle factors. It includes synthetic data generation, model training using a Random Forest Classifier, evaluation, and serialization.

## 📋 Features Included
- **Synthetic Data Generation**: Creates a mock healthcare dataset (`hypertension_data.csv`) including attributes like Age, BMI, Genetic Risk, Smoking status, and Salt Intake.
- **Data Preprocessing**: Handles train-test splitting and feature scaling via Standardisation.
- **Model Training**: Standard Random Forest Classifier implementation.
- **Model Deployment Ready**: Exports the trained classifier (`hypertension_model.pkl`) and the pre-configured scaler (`scaler.pkl`) using pickle.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/hypertension-prediction.git
cd hypertension-prediction
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Training Script
Execute the pipeline script to generate the synthetic data, train the model, and save the binary files:
```bash
python train.py
```

## 🛠️ Project Structure
```text
├── hypertension_data.csv  # Generated dataset (after running train.py)
├── hypertension_model.pkl # Saved model artifact (after running train.py)
├── scaler.pkl             # Saved feature scaler (after running train.py)
├── train.py               # Complete training and evaluation script
└── requirements.txt       # Python packages needed
```

---
*Disclaimer: This is for informational purposes only. For medical advice or diagnosis, consult a professional. AI responses may include mistakes.*
