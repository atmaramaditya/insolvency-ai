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
        # FAIL-SAFE PATH LOGIC
        # We check three possible locations for the assets
        possible_paths = [
            'deployment_assets/best_ensemble.pkl', # Relative to Root (Streamlit Cloud Default)
            os.path.join(os.path.dirname(__file__), 'deployment_assets/best_ensemble.pkl'), # Local relative
            '/mount/src/insolvency-ai/deployment_assets/best_ensemble.pkl' # Absolute Cloud Path
        ]
        
        feature_paths = [
            'deployment_assets/model_features.pkl',
            os.path.join(os.path.dirname(__file__), 'deployment_assets/model_features.pkl'),
            '/mount/src/insolvency-ai/deployment_assets/model_features.pkl'
        ]

        model_loaded = False
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    ml_assets["model"] = joblib.load(p)
                    logger.info(f"✅ Model loaded from: {p}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.error(f"Error loading pkl from {p}: {e}")

        features_loaded = False
        for fp in feature_paths:
            if os.path.exists(fp):
                try:
                    ml_assets["features"] = joblib.load(fp)
                    logger.info(f"✅ Features loaded from: {fp}")
                    features_loaded = True
                    break
                except Exception as e:
                    logger.error(f"Error loading pkl from {fp}: {e}")

        if not model_loaded or not features_loaded:
            logger.error("❌ Critical Error: Could not find deployment_assets in any known path.")
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

        # 2. Taiwan Dataset Fix: Neutral Imputation (0.5)
        input_df = pd.DataFrame(0.5, index=[0], columns=expected_features)

        # 3. Mapping logic (Taiwan Specific Names)
        mapping_targets = {
            "net_profit": [" Net Income to Total Assets", "Net Income to Total Assets"],
            "total_debt": [" Debt ratio %", "Debt ratio %"],
            "working_cap": [" Working Capital to Total Assets", "Working Capital to Total Assets"]
        }
        
        user_inputs = {"net_profit": net_profit, "total_debt": total_debt, "working_cap": working_cap}
        
        clean_expected = [str(col).strip() for col in expected_features]

        for input_key, variations in mapping_targets.items():
            for v in variations:
                v_clean = v.strip()
                if v_clean in clean_expected:
                    actual_col_name = expected_features[clean_expected.index(v_clean)]
                    input_df.at[0, actual_col_name] = user_inputs[input_key]
                    break 

        # 4. Final alignment
        input_df = input_df[expected_features]
        
        # 5. Inference
        probs = model.predict_proba(input_df)[0]
        risk_score = float(probs[1]) 
        
        prediction = 1 if risk_score > 0.5 else 0

        return {
            "status": "Bankrupt" if prediction == 1 else "Healthy",
            "risk_score": risk_score,
            "is_high_risk": bool(risk_score > 0.7)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
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
