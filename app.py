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
def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"text": "", "sentiment": None, "confidence": None}
    )

@app.post("/api/predict", response_class=HTMLResponse)
async def predict(request: Request):
    # Form থেকে পাঠানো সব ডাটা ধরা হচ্ছে
    form_data = await request.form()
    
    # HTML Form-এ নাম 'text', 'text_input' বা অন্য যাই থাক না কেন তা রিড করবে
    text = ""
    for key in form_data:
        if form_data[key]:
            text = form_data[key]
            break

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
