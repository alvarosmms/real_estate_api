import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import json
from PIL import Image

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title='Enterprise Real Estate',
    page_icon='🏙️',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# ---------- Estilo personalizado ----------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    label, .stSelectbox label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Cabecera ----------

import streamlit as st

# Imagen de cabecera
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("banner_enterprise_real_estate.png", use_container_width=True)
st.markdown("<hr style='border: 1px solid #eee; margin-top: 1rem;'></div>", unsafe_allow_html=True)


# ---------- Color de fondo ----------

st.markdown(
    """
    <style>
        body {
            background-color: #d4a55d;
        }
        .stApp {
            background-color: #d4a55d;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Mapa interactivo ----------
st.markdown("""
<h3 style='color: #000000;'>🗺️ Explora las zonas de Madrid</h3>
""", unsafe_allow_html=True)


with open('madrid_barrios_clean.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

m = folium.Map(location=[40.4168, -3.7038], zoom_start=11, tiles='cartodbpositron')

folium.GeoJson(
    geojson_data,
    name='Zonas de Madrid',
    tooltip=folium.GeoJsonTooltip(fields=['NOMBRE'], aliases=['Zona:']),
    highlight_function=lambda x: {'weight': 3, 'color': '#007BFF'},
).add_to(m)

st_data = st_folium(m, width=700, height=500)

# Obtener zonas del geojson
zonas_geojson = sorted({f['properties']['NOMBRE'] for f in geojson_data['features']})

# Detectar selección del mapa
selected_zona = None
if st_data and st_data.get('last_active_drawing'):
    props = st_data['last_active_drawing'].get('properties', {})
    selected_zona = props.get('NOMBRE')

# ---------- Inputs ----------
st.markdown("""
<h3 style='color: #000000;'> 📌 Selecciona la zona</h3>
""", unsafe_allow_html=True)

zona = st.selectbox('Zona de interés', zonas_geojson, index=zonas_geojson.index(selected_zona) if selected_zona in zonas_geojson else 0)
habitaciones = st.selectbox('🏢 Nº de habitaciones', list(range(0, 8)))
banos = st.selectbox('🛁 Nº de baños', list(range(0, 5)))
tipovivienda = st.selectbox('🏠 Tipo de vivienda', ['Piso', 'Ático', 'Chalet', 'Dúplex', 'Estudio', 'Otro'])
metros = st.number_input('📏 Metros cuadrados aproximados', min_value=10, max_value=1000, value=70)

# ---------- Predicción ----------
st.markdown('---')
if st.button('🔍 Predecir precio estimado'):
    params = {
        'zona': zona,
        'habitaciones': habitaciones,
        'banos': banos,
        'tipovivienda': tipovivienda,
        'metros': metros
    }

    try:
        response = requests.get('https://real-estate-api-22xe.onrender.com/predict', params=params)
        if response.status_code == 200:
            resultado = response.json()
            mensaje = resultado['mensaje']
            st.success(f'💰 {mensaje}')
        else:
            st.error('❌ No se pudo obtener la predicción. Inténtalo de nuevo más tarde.')
    except Exception as e:
        st.error(f'⚠️ Error al conectar con la API: {e}')

