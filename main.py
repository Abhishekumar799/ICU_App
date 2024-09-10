from typing import Union
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import numpy as np
from pycaret.classification import load_model, predict_model
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

app = FastAPI()

# Load the model
model = load_model('Model/lightgbm_Gosis_model')

# Load data for validation
data = pd.read_csv("Data/gosis-1-24hr.csv")
data.columns = list(map(str.strip, list(data.columns)))

# Set up template and static file directories
templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

def get_float(value: float, field_name: float):
    try:
        value = float(value)
        min_value = float(data.describe()[field_name].loc["min"])
        max_value = float(data.describe()[field_name].loc["max"])

        if value < min_value or value > max_value:
            raise ValueError(f"Please fill {field_name} in the range of {min_value} and {max_value}")
        return value
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("ICU_24hr.html", {"request": request})

@app.post("/submit")
async def submit(
    age: float = Form(...),
    height: float = Form(...),
    hospital_los_days: float = Form(...),
    icu_los_days: float = Form(...),
    weight: float = Form(...),
    bun_apache: float = Form(...),
    creatinine_apache: float = Form(...),
    gcs_eyes_apache: float = Form(...),
    glucose_apache: float = Form(...),
    heart_rate_apache: float = Form(...),
    hematocrit_apache: float = Form(...),
    map_apache: float = Form(...),
    resprate_apache: float = Form(...),
    sodium_apache: float = Form(...),
    temp_apache: float = Form(...),
    urineoutput_apache: float = Form(...),
    ventilated_apache: float = Form(...),
    wbc_apache: float = Form(...),
    d1_heartrate_max: float = Form(...),
    d1_heartrate_min: float = Form(...),
    d1_spo2_max: float = Form(...),
    d1_spo2_min: float = Form(...),
    d1_sysbp_max: float = Form(...),
    d1_sysbp_min: float = Form(...),
    h1_heartrate_max: float = Form(...),
    h1_heartrate_min: float = Form(...),
    h1_spo2_max: float = Form(...),
    h1_spo2_min: float = Form(...),
    h1_sysbp_max: float = Form(...),
    h1_sysbp_min: float = Form(...),
    d1_potassium_max: float = Form(...),
    d1_potassium_min: float = Form(...),
):
    data_dict = {
        'age': get_float(age, 'age'),
        'height': get_float(height, 'height'),
        'hospital_los_days': get_float(hospital_los_days, 'hospital_los_days'),
        'icu_los_days': get_float(icu_los_days, 'icu_los_days'),
        'weight': get_float(weight, 'weight'),
        'bun_apache': get_float(bun_apache, 'bun_apache'),
        'creatinine_apache': get_float(creatinine_apache, 'creatinine_apache'),
        'gcs_eyes_apache': get_float(gcs_eyes_apache, 'gcs_eyes_apache'),
        'glucose_apache': get_float(glucose_apache, 'glucose_apache'),
        'heart_rate_apache': get_float(heart_rate_apache, 'heart_rate_apache'),
        'hematocrit_apache': get_float(hematocrit_apache, 'hematocrit_apache'),
        'map_apache': get_float(map_apache, 'map_apache'),
        'resprate_apache': get_float(resprate_apache, 'resprate_apache'),
        'sodium_apache': get_float(sodium_apache, 'sodium_apache'),
        'temp_apache': get_float(temp_apache, 'temp_apache'),
        'urineoutput_apache': get_float(urineoutput_apache, 'urineoutput_apache'),
        'ventilated_apache': get_float(ventilated_apache, 'ventilated_apache'),
        'wbc_apache': get_float(wbc_apache, 'wbc_apache'),
        'd1_heartrate_max': get_float(d1_heartrate_max, 'd1_heartrate_max'),
        'd1_heartrate_min': get_float(d1_heartrate_min, 'd1_heartrate_min'),
        'd1_spo2_max': get_float(d1_spo2_max, 'd1_spo2_max'),
        'd1_spo2_min': get_float(d1_spo2_min, 'd1_spo2_min'),
        'd1_sysbp_max': get_float(d1_sysbp_max, 'd1_sysbp_max'),
        'd1_sysbp_min': get_float(d1_sysbp_min, 'd1_sysbp_min'),
        'h1_heartrate_max': get_float(h1_heartrate_max, 'h1_heartrate_max'),
        'h1_heartrate_min': get_float(h1_heartrate_min, 'h1_heartrate_min'),
        'h1_spo2_max': get_float(h1_spo2_max, 'h1_spo2_max'),
        'h1_spo2_min': get_float(h1_spo2_min, 'h1_spo2_min'),
        'h1_sysbp_max': get_float(h1_sysbp_max, 'h1_sysbp_max'),
        'h1_sysbp_min': get_float(h1_sysbp_min, 'h1_sysbp_min'),
        'd1_potassium_max': get_float(d1_potassium_max, 'd1_potassium_max'),
        'd1_potassium_min': get_float(d1_potassium_min, 'd1_potassium_min')
    }

    input_df = pd.DataFrame([data_dict])

    output, confidence = predict(model=model, input_df=input_df)
    if output == 1:
        return {"result": "Patient has a high chance of mortality", "confidence": confidence}
    else:
        return {"result": "Patient is fine", "confidence": confidence}

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['prediction_label'][0]
    confidence = predictions_df['prediction_score'][0]
    return predictions, confidence

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)