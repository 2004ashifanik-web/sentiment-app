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
        context={"request": request, "text": "", "sentiment": None, "confidence": None}
    )

@app.post("/api/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    
    # Form-এর যেকোনো কি (Key) থেকে মান নিয়ে নেওয়ার নিরাপদ পদ্ধতি
    text = ""
    for value in form.values():
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break

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
            "request": request,
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }
    )
