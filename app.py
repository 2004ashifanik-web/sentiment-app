import os
import io
import joblib
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "sentiment_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Model Load Error: {e}")

def get_sentiment_data(text: str):
    sentiment = "Unknown"
    confidence = 0.0
    
    if model and text.strip():
        try:
            pred = model.predict([text])[0]
            probs = model.predict_proba([text])[0]
            sentiment = str(pred)
            confidence = round(float(max(probs)) * 100, 2)
        except Exception as e:
            sentiment = f"Error: {str(e)}"
            
    return sentiment, confidence

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request, "text": "", "sentiment": None, "confidence": None}
    )

@app.post("/api/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    text = str(form_data.get("text", "")).strip()
    
    sentiment, confidence = get_sentiment_data(text)

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

@app.post("/api/bulk-predict", response_class=HTMLResponse)
async def bulk_predict(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    results = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    try:
        contents = await file.read()
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        text_column = df.columns[0]
        
        for idx, row in df.iterrows():
            txt = str(row[text_column])
            sent, conf = get_sentiment_data(txt)
            results.append({"text": txt, "sentiment": sent, "confidence": conf})
            
            s_lower = sent.lower()
            if "pos" in s_lower: pos_count += 1
            elif "neg" in s_lower: neg_count += 1
            else: neu_count += 1

    except Exception as e:
        results = [{"text": f"File Processing Error: {str(e)}", "sentiment": "Error", "confidence": 0}]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "bulk_results": results,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "neu_count": neu_count
        }
    )
