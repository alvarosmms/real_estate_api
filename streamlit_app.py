import streamlit as st
import pandas as pd
import requests

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Enterprise Real Estate",
    page_icon="🏙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Fondo personalizado ----------
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

.block-container {{
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------- Título ----------
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🏙️ Enterprise Real Estate</h1>
    <p style='text-align: center; color: gray;'>Predice el precio estimado de una vivienda en Madrid</p>
    <hr>
""", unsafe_allow_html=True)

# ---------- Cargar zonas ----------
@st.cache_data

def obtener_zonas():
    df = pd.read_csv("Datos_preprocesados.csv")
    return sorted(df["zona"].dropna().unique())

zonas = obtener_zonas()

# ---------- Inputs ----------
st.markdown("## Selecciona las características de la vivienda")
zona = st.selectbox("📍 Zona", zonas)
habitaciones = st.selectbox("🏢 Nº de habitaciones", [0 ,1, 2, 3, 4,5,6,7])
banos = st.selectbox("🛁 Nº de baños", [0, 1, 2, 3, 4])

# ---------- Predicción ----------
if st.button("🔍 Predecir precio estimado"):
    params = {
        "zona": zona,
        "habitaciones": habitaciones,
        "banos": banos
    }

    try:
        response = requests.get("https://real-estate-api-22xe.onrender.com/predict", params=params)
        if response.status_code == 200:
            resultado = response.json()
            precio = resultado["prediccion_precio"]
            st.success(f"💰 Precio estimado: **{precio:,.2f} €**")
        else:
            st.error("❌ Error al obtener la predicción. Inténtalo más tarde.")
    except Exception as e:
        st.error(f"⚠️ Error al conectar con la API: {e}")
