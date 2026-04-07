
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from config import MODEL_DIR

class Transaction(BaseModel):
    features: dict

app = FastAPI()

artifact = joblib.load(f"{MODEL_DIR}/lgbm_pipeline.joblib")
model = artifact['model']
preproc = artifact['preproc']
threshold = artifact['threshold']

@app.post('/score')
def score(tx: Transaction):
    import pandas as pd
    X = pd.DataFrame([tx.features])
    X_t = preproc.transform(X)
    score = float(model.predict_proba(X_t)[:, 1][0])
    label = int(score >= threshold)
    return {'score': score, 'fraud': label}
