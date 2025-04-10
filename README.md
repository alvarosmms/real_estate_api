# 🏡 Enterprise Real Estate

## 📌 Descripción

**Enterprise Real Estate** es un proyecto que combina un modelo de Machine Learning con una API RESTful construida en Flask y una interfaz visual elegante con Streamlit. Permite predecir el precio estimado de una vivienda en función de tres factores:

- Zona 🏙️
- Número de habitaciones 🛏️
- Número de baños 🚿
- Metros cuadrados 📐
---

## 🧠 Modelo de Machine Learning

Se utiliza **CatBoost**, un modelo avanzado de gradient boosting optimizado para variables categóricas. El modelo se entrena a partir de un dataset de viviendas y se guarda en un archivo `.pkl` que se carga tanto en la API como en Streamlit.

---

## 🌐 API REST

Desarrollada con **Flask**, la API ofrece un endpoint `/predict` donde se envían los parámetros de zona, habitaciones y baños por URL. Devuelve una predicción de precio en formato JSON.

---

## 🎨 Interfaz Streamlit

El archivo streamlit_app_2.0.py ofrece una interfaz profesional estilo inmobiliaria:

Desplegables personalizados con los valores reales del dataset.

Predicción en tiempo real a través de la API.

Visualización en mapa centrado en Madrid con Folium.

Estilo elegante con título y mensaje personalizados.

🔗 Demo online: https://realestateapi-lpyvtucdywsqevrowrjvqc.streamlit.app/

---

### 📁 Estructura del Proyecto

````plain text 
real_estate_api/
│
├── app.py                  # Código de la API Flask
├── streamlit_app_2.0.py    # Interfaz web mejorada con Streamlit y Folium
├── train_model.py          # Entrenamiento y guardado del modelo
├── Datos_preprocesados.csv # Dataset con datos de viviendas
├── requirements.txt        # Librerías necesarias
├── render.yaml             # Configuración para Render
└── README.md               # Este documento 😄
````

---

## ⚙️ Stack Usado

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-%20API-lightgrey?logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-brightgreen?logo=streamlit)
![Render](https://img.shields.io/badge/Render-Deployed-blueviolet?logo=render)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange?logo=catboost)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Array-yellow?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-lightgrey?logo=scikitlearn)
![Folium](https://img.shields.io/badge/Folium-Map-green?logo=python)

---