from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Sentiment Analysis API")
templates = Jinja2Templates(directory="templates")

# সেভ করা মডেল লোড করা
MODEL_PATH = "sentiment_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class TextPayload(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
def predict(payload: TextPayload):
    raw_text = payload.text.strip()
    
    if not raw_text:
        return {"text": "", "sentiment": "Neutral", "confidence": "0.0%"}

    if model is not None:
        prediction = model.predict([raw_text])[0]
        probabilities = model.predict_proba([raw_text])[0]
        
        max_prob = max(probabilities) * 100
        confidence_str = f"{max_prob:.1f}%"
    else:
        prediction = "Neutral"
        confidence_str = "50.0%"

    return {
        "text": raw_text,
        "sentiment": str(prediction),
        "confidence": confidence_str
    }