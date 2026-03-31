import logging
import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bankruptcy_api")

# --- SCHEMAS ---
class CompanyFinancials(BaseModel):
    net_profit_to_total_assets: float = Field(..., example=0.5)
    total_debt_to_total_assets: float = Field(..., example=0.2)
    working_capital_to_total_assets: float = Field(..., example=0.3)

class PredictionResponse(BaseModel):
    status: str
    risk_score: float
    is_high_risk: bool
    version: str

# --- ASSET LOADING ---
ml_assets = {"model": None, "features": None}

def load_assets():
    if ml_assets["model"] is None:
        try:
            logger.info("Loading ML assets...")
            # FIXED: Absolute pathing to prevent Errno 2 on Streamlit Cloud
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, 'deployment_assets', 'best_ensemble.pkl')
            feature_path = os.path.join(base_path, 'deployment_assets', 'model_features.pkl')
            
            ml_assets["model"] = joblib.load(model_path)
            ml_assets["features"] = joblib.load(feature_path)
            logger.info("Assets loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load assets: {e}")
            return False
    return True

def get_prediction(net_profit, total_debt, working_cap):
    if not load_assets():
        return None

    try:
        model = ml_assets["model"]
        
        # 1. Identify expected features
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = list(ml_assets["features"])

        # 2. FIXED: Initialize with 0.5 (Neutral) instead of 0.0
        # The Taiwan dataset features are largely ratios between 0 and 1. 
        # Filling 92 features with 0.0 makes the company look "non-existent" to the model.
        input_df = pd.DataFrame(0.5, index=[0], columns=expected_features)

        # 3. IMPROVED MAPPING LOGIC
        mapping_targets = {
            "net_profit": [
                " Net Income to Total Assets", 
                "Net Income to Total Assets", 
                "ROA(C) before interest and depreciation before interest"
            ],
            "total_debt": [
                " Debt ratio %", 
                "Debt ratio %", 
                "Total debt/Total net worth"
            ],
            "working_cap": [
                " Working Capital to Total Assets", 
                "Working Capital to Total Assets"
            ]
        }
        
        user_inputs = {"net_profit": net_profit, "total_debt": total_debt, "working_cap": working_cap}
        found_cols = []

        # Strip whitespace from expected features for cleaner matching
        clean_expected = [str(col).strip() for col in expected_features]

        for input_key, variations in mapping_targets.items():
            for v in variations:
                v_clean = v.strip()
                if v_clean in clean_expected:
                    actual_col_name = expected_features[clean_expected.index(v_clean)]
                    input_df.at[0, actual_col_name] = user_inputs[input_key]
                    found_cols.append(actual_col_name)
                    break 

        # 4. Final alignment
        input_df = input_df[expected_features]
        
        # 5. Inference
        probs = model.predict_proba(input_df)[0]
        risk_score = float(probs[1]) 
        
        prediction = 1 if risk_score > 0.5 else 0

        # --- DEBUG LOGGING ---
        print(f"\n--- DEBUG: Mapped: {found_cols} | Risk: {risk_score:.4f} ---")

        return {
            "status": "Bankrupt" if prediction == 1 else "Healthy",
            "risk_score": risk_score,
            "is_high_risk": bool(risk_score > 0.7)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- FASTAPI LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_assets()
    yield
    ml_assets.clear()

app = FastAPI(title="Bankruptcy Prediction API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def api_predict(financials: CompanyFinancials):
    res = get_prediction(
        financials.net_profit_to_total_assets,
        financials.total_debt_to_total_assets,
        financials.working_capital_to_total_assets
    )
    if res is None:
        raise HTTPException(status_code=500, detail="Inference Failed")
    
    return {**res, "version": "v8-production-ready"}
