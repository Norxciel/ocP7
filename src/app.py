import os

import streamlit as st

import requests as req

import numpy as np
import pandas as pd

import src.utils.funcs as funcs

from dotenv import load_dotenv
load_dotenv()

def request_API(client):
  print(client)
  if client is None:
    print("Select a client first")
    return
  
  BASE_URL = os.getenv("API_URL")
  url = f"{BASE_URL}/post_info"
  data = {"data": client.iloc[0].fillna('').to_dict()}

  # API call
  response = req.post(url, json=data)
  print(f"called url {response.url}")

  print(response)

  if response != 200:
    print(response.text)

  st.session_state.result = response.json().get("data")

if "test_data" not in st.session_state:
  print("test data not loaded. Loading...")
  _, test = funcs.prepare_data(*funcs.get_data_from_files())
  test = funcs.select_top_features(test, with_target=False)
  st.session_state.test_data = test
  print("Test data loaded!")

if "page" not in st.session_state:
    st.session_state.page = "Exemple"

st.sidebar.title("Menu")

page = st.sidebar.radio(label="", options=["Formulaire", "Affichage DataFrame"], label_visibility="hidden")

# if st.session_state.page == "formulaire":
if page == "Formulaire":
  st.title("Formulaire de prêt")

  with st.form("form"):
      nom = st.text_input("Nom")
      age = st.number_input("Âge", min_value=0, max_value=100, step=1)
      submitted = st.form_submit_button("Envoyer")

  if submitted:
      st.success(f"Nom: {nom}, Âge: {age}")

# elif st.session_state.page == "exemple":
elif page == "Affichage DataFrame":
  client = None

  if "client" not in st.session_state:
    st.session_state.client = st.session_state.test_data.sample(1)

  exemple_col1, exemple_col2 = st.columns([1, 1], gap="large")

  with exemple_col1:
    st.subheader("Données")

    if st.button("Autre client"):
      st.session_state.client = st.session_state.test_data.sample(1)

    st.write(st.session_state.client.T)

  with exemple_col2:
    st.subheader("Résultat")

    if st.button(
      "Requete",
      on_click=request_API,
      args=(st.session_state.client,)
    ):
      st.text(st.session_state.result if "result" in st.session_state else "Something went wrong...")
      st.write("Pret accordé" if ("result" in st.session_state and st.session_state.result < 0.23) else "Pret refusé")