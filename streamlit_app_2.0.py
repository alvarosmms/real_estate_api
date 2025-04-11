import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium
import json
from PIL import Image
from pathlib import Path

# Ruta donde se guardará el nuevo modelo
model_filename = "new_model.pkl"
model_path = Path("/mnt/data") / model_filename

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title='Enterprise RE',
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

# ----------  Descripción explicativa ---------- 
st.markdown("""
<p style='text-align: center; color: #000000; font-size: 1.1em; margin-top: -0.5rem;'>
    Selecciona en el mapa la zona en la que estés interesado.<br>
    O usa el menú desplegable para seleccionarla.<br>
    Después elige el número de habitaciones, de baños y los metros cuadrados aproximados.<br>
    Finalmente, pulsa el botón <strong>“Estimar precio”</strong> para obtener una valoración de tu futura vivienda.
</p>
<hr style='border: 1px solid #eee; margin-top: 1rem;'>
""", unsafe_allow_html=True)

# ---------- Mapa interactivo ----------
st.markdown("""
<h3 style='color: #000000;'>Explora el mapa de Madrid</h3>
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

zona = st.selectbox('📍 Zona de interés', zonas_geojson, index=zonas_geojson.index(selected_zona) if selected_zona in zonas_geojson else 0)
habitaciones = st.selectbox('🏢 Nº de habitaciones', list(range(0, 8)))
banos = st.selectbox('🛁 Nº de baños', list(range(0, 5)))
tipovivienda = st.selectbox('🏠 Tipo de vivienda', ['Piso', 'Ático', 'Chalet', 'Dúplex', 'Estudio', 'Otro'])
metros = st.number_input('📏 Metros cuadrados aproximados', min_value=10, max_value=1000, value=70)

# ---------- Predicción ----------
st.markdown('---')

# Centrar botón con HTML
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('🔍 Estimar precio'):
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

#NUEVO ALVARO
# ---------- Subida nuevo .csv ---------- 
st.markdown("---")
st.subheader("🔄 Subir nuevo dataset y reentrenar modelo")

uploaded_file = st.file_uploader("📁 Sube un nuevo archivo CSV con los datos", type=["csv"])

if uploaded_file:
    new_df = pd.read_csv(uploaded_file)

    required_columns = {"zona", "habitaciones", "banos", "tipovivienda", "metros", "precio"}
    if not required_columns.issubset(set(new_df.columns)):
        st.error(f"❌ El archivo debe contener las columnas: {', '.join(required_columns)}")
    else:
        from catboost import CatBoostRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error

        # Preparar datos
        new_df["zona"] = new_df["zona"].astype(str)
        new_df["tipovivienda"] = new_df["tipovivienda"].astype(str)

        X = new_df[["zona", "habitaciones", "banos", "tipovivienda", "metros"]]
        y = new_df["precio"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=7,
            cat_features=["zona", "tipovivienda"],
            eval_metric="RMSE",
            random_seed=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))

        # Guardar modelo
        import pickle
        with open("/mnt/data/new_model.pkl", "wb") as f:
            pickle.dump({"model": model, "mae": mae}, f)

        st.success("✅ Modelo reentrenado correctamente")

        # Enviar modelo a la API
        st.markdown("📤 ¿Quieres enviar el modelo a la API?")
        if st.button("🚀 Subir modelo a la API"):
            files = {"model_file": open("/mnt/data/new_model.pkl", "rb")}
            response = requests.post("https://real-estate-api-22xe.onrender.com/upload_model", files=files)
            if response.status_code == 200:
                st.success("🎉 Modelo actualizado en la API")
            else:
                st.error(f"❌ Error al subir el modelo: {response.text}")
#