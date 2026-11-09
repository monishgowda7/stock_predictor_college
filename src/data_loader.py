import os
import json
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "market_data_engineered.csv")

def load_dataset():
    """Load the preprocessed and feature-engineered dataset."""
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    return df

def load_models_and_metadata():
    """Load all trained models, scaler, and metadata."""
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    models = {
        "XGBoost (Tuned)": joblib.load(os.path.join(MODELS_DIR, "xgboost_tuned.pkl")),
        "CatBoost (Tuned)": joblib.load(os.path.join(MODELS_DIR, "catboost_tuned.pkl")),
        "LightGBM (Tuned)": joblib.load(os.path.join(MODELS_DIR, "lightgbm_tuned.pkl")),
        "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
    }
    
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    
    return models, scaler, metadata

def predict_direction(model_name, feature_dict, models, scaler, metadata):
    """
    Generate next-day direction prediction and probability given feature inputs.
    Returns: dict with 'prediction' (0 or 1), 'direction' ('UP' or 'DOWN'),
             'prob_up', 'prob_down', 'confidence'
    """
    selected_features = metadata["selected_features"]
    input_df = pd.DataFrame([feature_dict])[selected_features]
    
    model = models[model_name]
    
    if model_name == "Logistic Regression":
        X = pd.DataFrame(scaler.transform(input_df), columns=selected_features)
    else:
        X = input_df
        
    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    prob_down = float(probs[0])
    prob_up = float(probs[1])
    
    return {
        "prediction": pred,
        "direction": "UP (BULLISH)" if pred == 1 else "DOWN (BEARISH)",
        "prob_up": prob_up,
        "prob_down": prob_down,
        "confidence": prob_up if pred == 1 else prob_down
    }

def get_ensemble_prediction(feature_dict, models, scaler, metadata):
    """Generate consensus prediction across all models."""
    results = {}
    probs_up = []
    
    for name in models.keys():
        res = predict_direction(name, feature_dict, models, scaler, metadata)
        results[name] = res
        probs_up.append(res["prob_up"])
        
    avg_prob_up = float(np.mean(probs_up))
    avg_prob_down = 1.0 - avg_prob_up
    consensus_pred = 1 if avg_prob_up >= 0.5 else 0
    
    return {
        "consensus_direction": "UP (BULLISH)" if consensus_pred == 1 else "DOWN (BEARISH)",
        "avg_prob_up": avg_prob_up,
        "avg_prob_down": avg_prob_down,
        "individual_results": results
    }
