
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Predicción del Rendimiento Académico",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def cargar_modelos():
    modelo = tf.keras.models.load_model("modelo_rendimiento_academico.keras")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return modelo, encoder, scaler

modelo, encoder, scaler = cargar_modelos()

st.title("🎓 Predicción del Rendimiento Académico")
st.write("Estima el promedio académico a partir de características demográficas y educativas.")

st.subheader("Características del estudiante")

gender = st.selectbox("Género", ["female", "male"])
race = st.selectbox("Grupo étnico", [
    "group A", "group B", "group C", "group D", "group E"
])
education = st.selectbox("Nivel educativo de los padres", [
    "some high school", "high school",
    "some college", "associate's degree",
    "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Tipo de almuerzo", ["standard", "free/reduced"])
preparation = st.selectbox(
    "Curso de preparación",
    ["none", "completed"]
)

if st.button("Predecir rendimiento"):
    datos = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": preparation
    }])

    datos_encoded = encoder.transform(datos)
    datos_scaled = scaler.transform(datos_encoded)
    prediccion = modelo.predict(datos_scaled, verbose=0)[0][0]

    st.metric("Promedio académico estimado", f"{prediccion:.2f} / 100")
