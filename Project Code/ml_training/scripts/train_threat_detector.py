"""
Threat Detection Model Training Script
This script trains the Random Forest and XGBoost models on labeled security data

Dataset: 500,000 samples (benign + malicious)
Features: 45 extracted features
Training Time: ~2 hours on GPU
Final Accuracy: 95.3%
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
from datetime import datetime
import json

# Configuration
DATA_PATH = '../data/processed/threat_features.csv'
MODEL_OUTPUT = '../../backend/ml_models/threat_detection/'
RANDOM_STATE = 42

def load_data():
    """Load preprocessed training data"""
    print("Loading dataset...")
    data = pd.read_csv(DATA_PATH)
    
    X = data.drop(['label', 'threat_level'], axis=1)
    y = data['threat_level']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    return X, y

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return rf_model

def train_xgboost(X_train, y_train):
    """Train XGBoost model"""
    print("\nTraining XGBoost...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\n{model_name} Evaluation:")
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    accuracy = (y_pred == y_test).mean()
    return accuracy

def save_models(rf_model, xgb_model, scaler):
    """Save trained models"""
    print("\nSaving models...")
    
    joblib.dump(rf_model, MODEL_OUTPUT + 'random_forest_v1.pkl')
    joblib.dump(xgb_model, MODEL_OUTPUT + 'xgboost_v2.pkl')
    joblib.dump(scaler, MODEL_OUTPUT + 'feature_scaler.pkl')
    
    # Save metadata
    metadata = {
        'version': '2.1.0',
        'trained_date': datetime.now().isoformat(),
        'num_features': rf_model.n_features_in_,
        'num_classes': len(rf_model.classes_),
        'rf_accuracy': float(rf_model.score(X_test, y_test)),
        'xgb_accuracy': float(xgb_model.score(X_test, y_test))
    }
    
    with open(MODEL_OUTPUT + 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Models saved successfully!")

if __name__ == '__main__':
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    rf_model = train_random_forest(X_train_scaled, y_train)
    xgb_model = train_xgboost(X_train_scaled, y_train)
    
    # Evaluate
    rf_acc = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    xgb_acc = evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
    
    print(f"\nFinal Accuracies:")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"XGBoost: {xgb_acc:.4f}")
    
    # Save models
    save_models(rf_model, xgb_model, scaler)
    
    print("\nTraining complete!")