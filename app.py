from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model 
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def landing():
    return """
    <h1>API de Predicción de Precios de Viviendas 🏠</h1>
    <p>Usa el endpoint <code>/predict</code> con los siguientes parámetros:</p>
    <ul>
        <li><b>metros</b>: superficie en m2</li>
        <li><b>habitaciones</b>: número de habitaciones</li>
        <li><b>baños</b>: número de baños</li>
        <li><b>barrio</b>: zona o distrito codificado (por ahora, un número)</li>
    </ul>
    <p>Ejemplo: <code>/predict?metros=90&habitaciones=3&baños=2&barrio=5</code></p>
    """

@app.route("/predict", methods=["GET"])
def predict():
    try:
        metros = float(request.args.get("metros"))
        habitaciones = int(request.args.get("habitaciones"))
        baños = int(request.args.get("baños"))
        barrio = int(request.args.get("barrio"))

        X = np.array([[metros, habitaciones, baños, barrio]])
        pred = model.predict(X)[0]

        return jsonify({
            "precio_estimado": round(pred, 2),
            "input": {
                "metros": metros,
                "habitaciones": habitaciones,
                "baños": baños,
                "barrio": barrio
            }
        })
    except:
        return jsonify({"error": "Parámetros incorrectos o incompletos"}), 400

# Endpoint opcional preparado para redespliegue
# @app.route("/hello", methods=["GET"])
# def hello():
#     return "Hola, esto es un tercer endpoint activado tras redesplegar 🎉"

if __name__ == "__main__":
    app.run(debug=True)
