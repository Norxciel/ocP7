import os
import pickle

import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from typing import Dict, Any

class DataModel(BaseModel):
    data: Dict[str, Any]

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model_top.pickle")
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

@app.get("/")
def read_root():
    """ Test route.
    """
    return {"Hello": "World"}

@app.post("/post_info")
async def post_info(data: DataModel):
    user_info = data.data
    X_input = pd.DataFrame(user_info, index=[0])
    X_input = X_input.replace('', np.nan)

    print(f"{X_input.shape=}")
    print(f"{X_input=}")

    try:
        pred = model.predict_proba(X_input)[:, 1]
        return_data = pred.tolist()[0]

        response_data = {
            "data": return_data
        }

        status_code = 200

        return JSONResponse(content=response_data, status_code=status_code)
    except Exception as err:
        print(err)
        response_data = {
            "err": str(err)
        }
        return JSONResponse(content=response_data, status_code=500)