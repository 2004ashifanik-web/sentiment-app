import os
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "sentiment_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"text": "", "sentiment": None, "confidence": None}
    )

# UI এর সাথে মিল রাখতে রাউটটি /api/predict করা হয়েছে
@app.post("/api/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    sentiment = "Unknown"
    confidence = 0.0

    if model and text.strip():
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        sentiment = str(prediction)
        confidence = round(float(max(probabilities)) * 100, 2)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence
        }
    )
