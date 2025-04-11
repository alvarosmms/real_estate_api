from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from catboost import CatBoostRegressor
from collections import OrderedDict

# Mensaje con la predicción
def generar_mensaje_precio(prediccion, mae, confianza="95%"):
    min_price = round(prediccion - 0.1 * mae)
    max_price = round(prediccion + 0.1 * mae)
    mensaje = (
        f"🧭 Estimación de precio para tu vivienda:\n\n"
        f"📌 Con una confianza aproximada del {confianza * 100:.0f}%, se estima que el precio adecuado "
        f"se encuentra entre **{min_price:.0f} €** y **{max_price:.0f} €**.\n\n"
        f"ℹ️ Ten en cuenta que este intervalo depende de los datos proporcionados. El precio más ajustado puede variar en función "
        f"de características adicionales no incluidas, como:\n"
        f"- La localización exacta dentro de la zona\n"
        f"- La planta del inmueble\n"
        f"- La orientación y luminosidad\n"
        f"- Servicios como ascensor, calefacción, zonas comunes, etc.\n\n"
        f"🏡 Estos factores pueden influir significativamente en la valoración final."
    )
    return mensaje

app = Flask(__name__)

# Ruta del modelo
MODEL_PATH = "model.pkl"

# ========== Cargar modelo al iniciar ==========
try:
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        mae = data["mae"]
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None
    mae = None

#NUEVO ALVARO
# ========== Endpoint para actualizar modelo ==========
@app.route("/update-model", methods=["POST"])
def update_model():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file.save(MODEL_PATH)

    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            global model, mae
            model = data["model"]
            mae = data["mae"]
        return jsonify({"success": "Modelo actualizado correctamente"}), 200
    except Exception as e:
        return jsonify({"error": f"No se pudo cargar el modelo: {str(e)}"}), 500
    
# ========== Endpoint para recibir el modelo actualizado ==========
@app.route("/upload_model", methods=["POST"])
def upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({"error": "No se encontró ningún archivo en la solicitud"}), 400

        file = request.files['model_file']
        if file.filename == '':
            return jsonify({"error": "Nombre de archivo vacío"}), 400

        file.save("model.pkl")  # Sobrescribe el modelo actual

        # Recargar modelo tras guardarlo
        global model, mae
        with open("model.pkl", "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            mae = data["mae"]

        return jsonify({"message": "✅ Modelo actualizado exitosamente"}), 200

    except Exception as e:
        return jsonify({"error": f"Error al cargar el nuevo modelo: {str(e)}"}), 500

#

# ========== Rutas ==========
@app.route("/")
def home():
    return  """
        <h2>API para Predicción de Precio de viviendas en Madrid 🏡</h2>
        <p>Usa el endpoint <code>/predict</code> con los siguientes parámetros:</p>
        <ul>
            <li><b>zona</b> (str) - nombre del barrio o distrito</li>
            <li><b>habitaciones</b> (int) - número de habitaciones</li>
            <li><b>banos</b> (int) - número de baños</li>
            <li><b>tipovivienda</b> (str) - tipo de vivienda (Piso, Ático, Chalet...)</li>
            <li><b>metros</b> (int) - superficie en metros cuadrados</li>
        </ul>
        <p><b>Ejemplo:</b></p>
        <code>/predict?zona=Chamberí&habitaciones=3&banos=2&tipovivienda=Piso&metros=90</code>
    """

@app.route("/predict", methods=["GET"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    zona = request.args.get("zona")
    habitaciones = request.args.get("habitaciones", type=int)
    banos = request.args.get("banos", type=int)
    tipovivienda = request.args.get("tipovivienda")
    metros = request.args.get("metros", type=int)
    

    if not zona or habitaciones is None or banos is None or tipovivienda is None or metros is None:
        return jsonify({"error": "Parámetros incompletos"}), 400

    try:
        # Crear DataFrame
        input_data = pd.DataFrame([{
            "zona": zona,
            "habitaciones": habitaciones,
            "banos": banos,
            "tipovivienda": tipovivienda,
            "metros": metros
        }])

        # Asegurar tipo string en zona (si hace falta)
        input_data["zona"] = input_data["zona"].astype(str)
        input_data["tipovivienda"] = input_data["tipovivienda"].astype(str)

        prediction = model.predict(input_data)[0]
        mensaje = generar_mensaje_precio(prediction, mae, confianza=0.95)

        return jsonify({
            "zona": zona,
            "habitaciones": habitaciones,
            "banos": banos,
            "tipovivienda": tipovivienda,
            "metros": metros,
            "prediccion_precio": round(prediction, 2),
            "mensaje": mensaje
        })

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
