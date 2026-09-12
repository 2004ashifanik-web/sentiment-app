import os
import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "sentiment_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"text": "", "sentiment": None, "confidence": None}
    )

@app.post("/api/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    
    # Form data থেকে ডাটা নেওয়া হচ্ছে
    text = form_data.get("text", "").strip()

    sentiment = "Unknown"
    confidence = 0.0

    if model and text:
        try:
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            
            sentiment = str(prediction)
            confidence = round(float(max(probabilities)) * 100, 2)
        except Exception as e:
            sentiment = f"Error: {str(e)}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }
    )
