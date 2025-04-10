from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import os

app = Flask(__name__)

# Ruta del modelo
MODEL_PATH = "/Users/alvarosanchez/ONLINE_DS_THEBRIDGE_ALVAROSMMS-1/real_estate_api/Modelo/model.pkl"

# ========== Cargar modelo al iniciar ==========
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except:
    model = None

# ========== Landing page ==========
@app.route("/", methods=["GET"])
def landing():
    return """
    <h1>API de Predicción de Viviendas 🏡</h1>
    <p>Endpoints disponibles:</p>
    <ul>
        <li><b>/predict</b>: devuelve predicción con parámetros (GET)</li>
        <li><b>/retrain</b>: reentrena el modelo usando new_data.csv (POST)</li>
    </ul>
    """

# ========== Predicción ==========
@app.route("/predict", methods=["GET"])
def predict():
    if model is None:
        return jsonify({"error": "El modelo no está cargado"}), 500

    try:
        zona = request.args.get("zona")
        habitaciones = int(request.args.get("habitaciones"))
        baños = int(request.args.get("baños"))

        # Preprocesado simple
        input_data = pd.DataFrame([{
            "zona": zona,
            "habitaciones": habitaciones,
            "baños": baños
        }])
        input_data['zona'] = input_data['zona'].astype(str)

        pred = model.predict(input_data)[0]

        return jsonify({
            "precio_estimado": round(pred, 2),
            "input": {
                "zona": zona,
                "habitaciones": habitaciones,
                "baños": baños
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ========== Retrain Juan ==========
# @app.route("/retrain", methods=["POST"])
# def retrain():
#     try:
#         df = pd.read_csv("new_data.csv")
#         expected_cols = {'zona', 'precio', 'habitaciones', 'baños'}

#         if not expected_cols.issubset(set(df.columns)):
#             return jsonify({"error": "Faltan columnas necesarias"}), 400

#         df['zona'] = df['zona'].astype(str)
#         X = df[['zona', 'habitaciones', 'baños']]
#         y = df['precio']
#         cat_features = ['zona']

#         new_model = CatBoostRegressor(iterations=200, learning_rate=0.1, depth=7, verbose=0)
#         new_model.fit(X, y, cat_features=cat_features)

#         # Guardar el nuevo modelo
#         with open(MODEL_PATH, 'wb') as f:
#             pickle.dump(new_model, f)

#         global model
#         model = new_model  # Actualiza el modelo en memoria

#         return jsonify({"message": "Modelo reentrenado y guardado correctamente."}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# ========== Retrain Álvaro ==========

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        data = request.get_json()

        # Validación de entrada
        required_fields = {"zona", "precio", "habitaciones", "baños"}
        if not data or not required_fields.issubset(data.keys()):
            return jsonify({"error": f"Debes enviar: {required_fields}"}), 400

        # Crear DataFrame con un único registro
        df = pd.DataFrame([{
            "zona": str(data["zona"]),
            "precio": float(data["precio"]),
            "habitaciones": int(data["habitaciones"]),
            "baños": int(data["baños"])
        }])

        X = df[["zona", "habitaciones", "baños"]]
        y = df["precio"]
        cat_features = ["zona"]

        # Reentrenar con ese único registro (esto es una demo)
        new_model = CatBoostRegressor(iterations=10, learning_rate=0.5, depth=3, verbose=0)
        new_model.fit(X, y, cat_features=cat_features)

        # Guardar modelo actualizado
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)

        global model
        model = new_model

        return jsonify({"message": "Modelo reentrenado con el nuevo dato"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== Para redespliegue en clase==========
# @app.route("/hello", methods=["GET"])
# def hello():
#     return "Hola, este es el endpoint opcional para redespliegue 🛠️"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
