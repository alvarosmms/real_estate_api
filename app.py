from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from catboost import CatBoostRegressor
from collections import OrderedDict

app = Flask(__name__)

# Ruta del modelo
MODEL_PATH = "model.pkl"

# ========== Cargar modelo al iniciar ==========
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# ========== Rutas ==========
@app.route("/")
def home():
    return (
        "<h2>API para Predicción de Precio de Viviendas 🏡 </h2>"
        "<p>Usa el endpoint <code>/predict</code> con los parámetros:</p>"
        "<ul>"
        "<li><b>zona</b> (str)</li>"
        "<li><b>habitaciones</b> (int)</li>"
        "<li><b>banos</b> (int)</li>"
        "</ul>"
        "<p>Ejemplo: <code>/predict?zona=Chamberí&habitaciones=3&banos=2</code></p>"
    )

@app.route("/predict", methods=["GET"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    zona = request.args.get("zona")
    habitaciones = request.args.get("habitaciones", type=int)
    banos = request.args.get("banos", type=int)

    if not zona or habitaciones is None or banos is None:
        return jsonify({"error": "Parámetros incompletos"}), 400

    try:
        # Crear DataFrame
        input_data = pd.DataFrame([{
            "zona": zona,
            "habitaciones": habitaciones,
            "banos": banos
        }])

        # Asegurar tipo string en zona (si hace falta)
        input_data["zona"] = input_data["zona"].astype(str)

        prediction = model.predict(input_data)[0]

        return jsonify({
            "zona": zona,
            "habitaciones": habitaciones,
            "banos": banos,
            "prediccion_precio": round(prediction, 2)
        })
    
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Error al realizar la predicción: {str(e)}"}), 500


# ========== Para redespliegue en clase==========
# @app.route("/hello", methods=["GET"])
# def hello():
#     return "Hola, este es el endpoint opcional para redespliegue 🛠️"


# ========== Lanzar servidor ==========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
