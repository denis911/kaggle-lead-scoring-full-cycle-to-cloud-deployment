from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from typing import Optional

# --- Configuration ---
MODEL_PATH = "lead_scoring_model.joblib"
THRESHOLD = 0.4817

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Lead Scoring API", version="1.0")

# Load model pipeline
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Could not load model at {MODEL_PATH}")

from pydantic import BaseModel, Field

class LeadData(BaseModel):
    Lead_Origin: str = Field(alias="Lead Origin")
    Lead_Source: Optional[str] = Field("Unknown", alias="Lead Source")
    Do_Not_Email: str = Field(alias="Do Not Email")
    TotalVisits: float
    Total_Time_Spent_on_Website: float = Field(alias="Total Time Spent on Website")
    Page_Views_Per_Visit: float = Field(alias="Page Views Per Visit")
    Last_Activity: Optional[str] = Field("Unknown", alias="Last Activity")
    Country: Optional[str] = Field("Unknown")
    Specialization: Optional[str] = Field("Unknown")
    What_is_your_current_occupation: Optional[str] = Field("Unknown", alias="What is your current occupation")
    City: Optional[str] = Field("Unknown")
    A_free_copy_of_Mastering_The_Interview: str = Field(alias="A free copy of Mastering The Interview")
    Last_Notable_Activity: str = Field(alias="Last Notable Activity")

    model_config = {
        "populate_by_name": True
    }

def consolidate_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate the consolidation logic from train.py to ensure parity."""
    if 'Lead Source' in df.columns:
        df['Lead Source'] = df['Lead Source'].replace(['google'], 'Google')
        # Note: In a real prod app, 'rare' sources should be matched against 
        # the specific list from training. For now, we apply the general logic.
        # But handle_unknown='ignore' in OneHotEncoder will handle novel rare types.
        
    return df

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: LeadData):
    try:
        # 1. Convert input to DataFrame with correct column names
        input_dict = {
            "Lead Origin": data.Lead_Origin,
            "Lead Source": data.Lead_Source,
            "Do Not Email": data.Do_Not_Email,
            "TotalVisits": data.TotalVisits,
            "Total Time Spent on Website": data.Total_Time_Spent_on_Website,
            "Page Views Per Visit": data.Page_Views_Per_Visit,
            "Last Activity": data.Last_Activity,
            "Country": data.Country,
            "Specialization": data.Specialization,
            "What is your current occupation": data.What_is_your_current_occupation,
            "City": data.City,
            "A free copy of Mastering The Interview": data.A_free_copy_of_Mastering_The_Interview,
            "Last Notable Activity": data.Last_Notable_Activity
        }
        df = pd.DataFrame([input_dict])

        # 2. Consolidate (Pre-processing)
        df = consolidate_categories(df)

        # 3. Predict
        probs_raw = model.predict_proba(df)[:, 1][0]
        probs = float(probs_raw)
        prediction = int(probs >= THRESHOLD)
        
        return {
            "lead_score": round(probs * 100, 2),
            "conversion_probability": round(probs, 4),
            "is_hot_lead": bool(prediction),
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
