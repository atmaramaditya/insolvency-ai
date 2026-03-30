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

# --- ASSET LOADING LOGIC ---
ml_assets = {"model": None, "features": None}

def load_assets():
    if ml_assets["model"] is None:
        try:
            logger.info("Loading ML assets...")
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
        
        # Determine features the model expects
        try:
            expected_features = list(model.feature_names_in_)
        except AttributeError:
            expected_features = list(ml_assets["features"])

        # 1. Create empty DataFrame
        input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)

        # --- CORRECTED INDENTATION SECTION ---
        mapping_targets = {
            "net_profit": [
                " Net Income to Total Assets",         # Added variation 1
                "Net Income to Total Assets",          # Added variation 2
                " Net income to total assets",         # Added variation 3
                "Net income to total assets",          # Added variation 4
                "ROA(C) before interest and depreciation before interest", # Common in UCI dataset
                " Net Value Per Share (A)"             # Another common one
            ],
            "total_debt": [" Debt ratio %", "Debt ratio %"],
            "working_cap": [" Working Capital to Total Assets", "Working Capital to Total Assets"]
        }
        
        user_inputs = {"net_profit": net_profit, "total_debt": total_debt, "working_cap": working_cap}
        
        # Try to find which specific variation exists in your actual model
        found_cols = []
        for input_key, variations in mapping_targets.items():
            for v in variations:
                if v in expected_features:
                    input_df.loc[0, v] = user_inputs[input_key]
                    found_cols.append(v)
                    break 

        # 3. Final alignment
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # 4. Inference
        # We use predict_proba to get the continuous risk score
        probability = float(model.predict_proba(input_df)[0][1])
        prediction = 1 if probability > 0.5 else 0

        # --- TERMINAL DEBUG (ADITYA: CHECK THIS) ---
        print("\n" + "="*40)
        print("🔍 ADITYA'S MODEL DEBUGGER")
        print(f"Mapped Columns Found: {found_cols}")
        print(f"Slider Values -> NP: {net_profit}, Debt: {total_debt}, WC: {working_cap}")
        print(f"FINAL RISK SCORE: {round(probability * 100, 2)}%")
        print("="*40 + "\n")

        return {
            "status": "Bankrupt" if prediction == 1 else "Healthy",
            "risk_score": probability,
            "is_high_risk": bool(probability > 0.7)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return None

# --- FASTAPI WRAPPER ---
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
    
    return {**res, "version": "v7-robust-mapping"}