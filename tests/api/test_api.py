import pytest

import pickle

import numpy as np
import pandas as pd

from fastapi.testclient import TestClient

def test_read_main(mocker):
    mocker.patch("builtins.open")
    mocker.patch("pickle.load")
    
    import src.api as api
    client = TestClient(api.app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_posting_data(mocker):
    # Mocking File reading
    mocker.patch("builtins.open")
    mocker.patch("pickle.load")

    import src.api as api
    client = TestClient(api.app)

    # Mocking model prediction
    mocked_model = mocker.MagicMock()
    mocked_model.predict_proba.return_value = np.array([[0.1, 0.22]])
    mocker.patch("src.api.model", mocked_model)

    # Fake data for POST
    df = pd.DataFrame({'EXT_SOURCE_3': '', 'OBS_30_CNT_SOCIAL_CIRCLE': 2.0, 'OBS_60_CNT_SOCIAL_CIRCLE': 2.0, 'EXT_SOURCE_2': 0.6326323416559466, 'AMT_REQ_CREDIT_BUREAU_YEAR': 2.0, 'CNT_CHILDREN': 0.0, 'CNT_FAM_MEMBERS': 1.0, 'CREDIT_TERM': 0.07160685375067458, 'NONLIVINGAREA_MEDI': 0.1983, 'EXT_SOURCE_1': 0.7974448949384467, 'DAYS_LAST_PHONE_CHANGE': -2873.0, 'NONLIVINGAREA_MODE': 0.2057, 'AMT_GOODS_PRICE': 238500.0, 'REGION_POPULATION_RELATIVE': 0.02461, 'NONLIVINGAREA_AVG': 0.1943}, index=[0])
    data = {"data": df.iloc[0].fillna('').to_dict()}

    response = client.post("/post_info", json=data)

    assert response.status_code == 200
    assert 'data' in response.json()
    assert response.json().get('data') == 0.22