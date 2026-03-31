import logging
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# --- 1. LOGGING & GLOBAL ASSETS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bankruptcy_api")

# Global dictionary to store the model and features
ml_assets = {"model": None, "features": None}

# --- 2. ASSET LOADING (Defined BEFORE get_prediction) ---
def load_assets():
    """Loads ML models if they are not already in memory."""
    if ml_assets["model"] is not None:
        return True

    # Check multiple possible paths (Local vs. Streamlit Cloud)
    base_dir = os.path.dirname(__file__)
    model_paths = [
        'deployment_assets/best_ensemble.pkl',
        os.path.join(base_dir, 'deployment_assets/best_ensemble.pkl'),
    ]
    
    feature_paths = [
        'deployment_assets/model_features.pkl',
        os.path.join(base_dir, 'deployment_assets/model_features.pkl'),
    ]

    try:
        # Load Model
        for p in model_paths:
            if os.path.exists(p):
                ml_assets["model"] = joblib.load(p)
                logger.info(f"✅ Model loaded from: {p}")
                break
        
        # Load Feature List
        for fp in feature_paths:
            if os.path.exists(fp):
                ml_assets["features"] = joblib.load(fp)
                logger.info(f"✅ Features loaded from: {fp}")
                break

    except Exception as e:
        logger.error(f"❌ Critical loading error: {e}")
        return False

    return ml_assets["model"] is not None

# --- 3. PREDICTION LOGIC ---
def get_prediction(net_profit, total_debt, working_cap, custom_data: Optional[pd.DataFrame] = None):
    # This call now works because load_assets is defined above
    if not load_assets():
        return None

    try:
        model = ml_assets["model"]
        # Handle different model types (XGBoost vs Sklearn)
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = list(ml_assets["features"])

        # Create base dataframe with neutral imputation (Taiwan Dataset standard is often 0.5 for scaled data)
        if custom_data is not None:
            input_df = custom_data.reindex(columns=expected_features, fill_value=0.5)
        else:
            input_df = pd.DataFrame(0.5, index=[0], columns=expected_features)
            
            # Mapping logic with the exact Taiwan Dataset Column Strings
            # NOTE: These names MUST match your training data exactly (including spaces)
            mapping = {
                " Net Income to Total Assets": net_profit,
                " Debt ratio %": total_debt,
                " Working Capital to Total Assets": working_cap
            }
            
            for col, val in mapping.items():
                if col in expected_features:
                    input_df.at[0, col] = val
                elif col.strip() in [c.strip() for c in expected_features]:
                    # Fallback for slight naming mismatches
                    actual_col = [c for c in expected_features if c.strip() == col.strip()][0]
                    input_df.at[0, actual_col] = val

        # Ensure correct column order
        input_df = input_df[expected_features]

        # INFERENCE
        probs = model.predict_proba(input_df)
        risk_scores = probs[:, 1]
        
        # Explanation logic for the dashboard
        feature_importance = {
            "Profitability": net_profit,
            "Leverage": total_debt,
            "Liquidity": working_cap
        }

        return {
            "status": "Bankrupt" if risk_scores[0] > 0.5 else "Healthy",
            "risk_score": float(risk_scores[0]),
            "all_scores": risk_scores.tolist(), # For batch processing
            "explanations": feature_importance
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return None

# --- 4. FASTAPI SETUP (For API Deployment) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield
    ml_assets.clear()

app = FastAPI(title="Insolvency AI API", lifespan=lifespan)

@app.get("/")
def health_check():
    return {"status": "operational", "model_loaded": ml_assets["model"] is not None}
