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

class LeadData(BaseModel):
    Lead_Origin: str
    Lead_Source: Optional[str] = "Unknown"
    Do_Not_Email: str
    TotalVisits: float
    Total_Time_Spent_on_Website: float
    Page_Views_Per_Visit: float
    Last_Activity: Optional[str] = "Unknown"
    Country: Optional[str] = "Unknown"
    Specialization: Optional[str] = "Unknown"
    What_is_your_current_occupation: Optional[str] = "Unknown"
    City: Optional[str] = "Unknown"
    A_free_copy_of_Mastering_The_Interview: str
    Last_Notable_Activity: str

    class Config:
        # Map snake_case or spaces to the exact model feature names
        fields = {
            "Lead_Origin": "Lead Origin",
            "Lead_Source": "Lead Source",
            "Do_Not_Email": "Do Not Email",
            "Total_Time_Spent_on_Website": "Total Time Spent on Website",
            "Page_Views_Per_Visit": "Page Views Per Visit",
            "Last_Activity": "Last Activity",
            "What_is_your_current_occupation": "What is your current occupation",
            "A_free_copy_of_Mastering_The_Interview": "A free copy of Mastering The Interview",
            "Last_Notable_Activity": "Last Notable Activity"
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
