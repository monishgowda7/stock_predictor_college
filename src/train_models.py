import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def train_and_export_models():
    os.makedirs("models", exist_ok=True)
    
    # 1. Load data
    data_path = os.path.join("data", "market_data_engineered.csv")
    print(f"Loading data from {data_path}...")
    market_df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    
    # Selected 25 features as identified in notebook
    selected_features = [
        'SP500_Return', 'NASDAQ_Return', 'DOW_Return', 'NIFTY_Return', 
        'USDINR_Return', 'CRUDE_Return', 'VIX_Return', 'NASDAQ_Volume', 
        'SP500_Volume', 'CRUDE_Volume', 'DOW_Volume', 'Rolling_Volatility_20', 
        'ROC_10', 'MACD_Signal', 'MACD_Histogram', 'Price_vs_SMA10', 
        'RSI_14', 'Price_vs_BB', 'Price_vs_EMA10', 'MACD', 
        'ATR_Percent', 'ATR_14', 'Price_vs_EMA20', 'Price_vs_SMA20', 'VIX_Low'
    ]
    
    # Chronological Split
    train_df = market_df.loc[:"2022-12-30"]
    val_df = market_df.loc["2023-01-03":"2024-12-31"]
    test_df = market_df.loc["2025-01-02":]
    
    print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    X_train = train_df[selected_features]
    y_train = train_df["Target"]
    
    X_val = val_df[selected_features]
    y_val = val_df["Target"]
    
    X_test = test_df[selected_features]
    y_test = test_df["Target"]
    
    # Standard Scaler (fitted on train only)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=selected_features, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features, index=X_test.index)
    
    models = {}
    
    # XGBoost Tuned (Version 3)
    print("Training XGBoost...")
    xgb_v3 = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_v3.fit(X_train, y_train)
    models["XGBoost (Tuned)"] = (xgb_v3, False)
    
    # CatBoost Tuned (Version 3)
    print("Training CatBoost...")
    catboost_v3 = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=7,
        random_seed=42,
        loss_function="Logloss",
        verbose=0
    )
    catboost_v3.fit(X_train, y_train)
    models["CatBoost (Tuned)"] = (catboost_v3, False)
    
    # LightGBM Tuned (Version 3)
    print("Training LightGBM...")
    lgbm_v3 = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgbm_v3.fit(X_train, y_train)
    models["LightGBM (Tuned)"] = (lgbm_v3, False)
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models["Random Forest"] = (rf_model, False)
    
    # Logistic Regression
    print("Training Logistic Regression...")
    logistic_v3 = LogisticRegression(
        C=0.01,
        solver="lbfgs",
        max_iter=3000,
        random_state=42
    )
    logistic_v3.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = (logistic_v3, True) # Needs scaling
    
    # Evaluate all models on test set
    metrics_summary = {}
    test_predictions = {}
    test_probabilities = {}
    
    for name, (mdl, needs_scale) in models.items():
        X_eval = X_test_scaled if needs_scale else X_test
        preds = mdl.predict(X_eval)
        probs = mdl.predict_proba(X_eval)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds).tolist()
        
        metrics_summary[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
            "ROC-AUC": round(float(roc), 4),
            "ConfusionMatrix": cm
        }
        test_predictions[name] = preds.tolist()
        test_probabilities[name] = probs.tolist()
        print(f"[{name}] Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC: {roc:.4f}")
        
        # Save individual model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(mdl, os.path.join("models", f"{clean_name}.pkl"))
        
    # Save scaler
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    
    # Feature importances for tree models
    importances = {
        "XGBoost": dict(zip(selected_features, [float(x) for x in xgb_v3.feature_importances_])),
        "Random Forest": dict(zip(selected_features, [float(x) for x in rf_model.feature_importances_]))
    }
    
    metadata = {
        "selected_features": selected_features,
        "metrics": metrics_summary,
        "feature_importances": importances,
        "test_dates": test_df.index.strftime("%Y-%m-%d").tolist(),
        "test_actuals": y_test.tolist(),
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities
    }
    
    with open(os.path.join("models", "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("\nAll models and metadata successfully exported to /models directory!")

if __name__ == "__main__":
    train_and_export_models()
