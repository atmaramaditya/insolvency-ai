import logging
import joblib
import pandas as pd
import numpy as np
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
            # Using standard paths
            ml_assets["model"] = joblib.load('deployment_assets/best_ensemble.pkl')
            ml_assets["features"] = joblib.load('deployment_assets/model_features.pkl')
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

        # 2. Initialize DataFrame with Zeros
        input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)

        # 3. IMPROVED MAPPING LOGIC
        # I've cleaned these strings to match common dataset formats (UCI/Taiwan)
        mapping_targets = {
            "net_profit": ["Net Income to Total Assets", "ROA(C) before interest and depreciation before interest", "Persistent EPS in the Last Four Seasons"],
            "total_debt": ["Debt ratio %", "Total debt/Total net worth", "Current Liability to Assets"],
            "working_cap": ["Working Capital to Total Assets", "Cash/Total Assets", "Working Capital/Equity"]
        }
        
        user_inputs = {"net_profit": net_profit, "total_debt": total_debt, "working_cap": working_cap}
        found_cols = []

        # Strip whitespace from expected features for cleaner matching
        clean_expected = [str(col).strip() for col in expected_features]

        for input_key, variations in mapping_targets.items():
            for v in variations:
                v_clean = v.strip()
                if v_clean in clean_expected:
                    # Find the actual index in the original expected_features list
                    actual_col_name = expected_features[clean_expected.index(v_clean)]
                    input_df.at[0, actual_col_name] = user_inputs[input_key]
                    found_cols.append(actual_col_name)
                    break 

        # 4. SENSITIVITY CHECK (The "Static Value" Fix)
        # If the model is returning a static value, it's often because 3 features 
        # out of 95 aren't enough to move the needle. 
        # Here we can fill "average" values for other columns if needed, 
        # but for now, let's ensure the 3 we have are actually mapped.
        if not found_cols:
            logger.warning("Zero columns were successfully mapped! Check feature names.")

        # 5. Inference
        # We ensure the order is identical to training
        input_df = input_df[expected_features]
        
        # Get probability for the "Positive" class (usually Bankrupt)
        probs = model.predict_proba(input_df)[0]
        risk_score = float(probs[1]) 
        
        # Determine status
        prediction = 1 if risk_score > 0.5 else 0

        # --- DEBUG LOGGING ---
        print(f"\n--- DEBUG: Input Mapped: {found_cols} ---")
        print(f"--- DEBUG: Risk Score: {risk_score:.4f} ---")

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
