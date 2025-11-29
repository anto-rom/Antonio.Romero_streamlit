import pickle
import numpy as np
import streamlit as st


# Cargar el modelo

@st.cache_resource
def load_model():
    with open("src/modelo_arbol_diabetes.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()


# Config básica de la página

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Predicción de riesgo de diabetes")
st.write("Aplicación de ejemplo utilizando un modelo de árbol de decisión entrenado previamente.")

st.markdown("---")


# Inputs del usuario

st.subheader("Introduce los datos del paciente")

# Ajusta estos campos a las columnas reales de tu modelo
pregnancies = st.number_input("Número de embarazos (Pregnancies)", min_value=0, max_value=20, value=1)
glucose = st.number_input("Nivel de glucosa (Glucose)", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Presión diastólica (BloodPressure)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Espesor de piel (SkinThickness)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulina (Insulin)", min_value=0, max_value=900, value=80)
bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
dpf = st.number_input("DiabetesPedigreeFunction (DPF)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.number_input("Edad (Age)", min_value=1, max_value=120, value=35)

# Crear el vector de features en el orden correcto
features = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])

st.markdown("---")


# Botón de predicción

if st.button("Predecir riesgo"):
    # Predicción
    pred = model.predict(features)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0][1]

    if pred == 1:
        st.error("🔴 El modelo predice **riesgo de diabetes**.")
    else:
        st.success("🟢 El modelo predice **bajo riesgo de diabetes**.")

    if prob is not None:
        st.write(f"Probabilidad estimada de diabetes: **{prob:.2%}**")
