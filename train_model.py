import pandas as pd
import pickle
from catboost import CatBoostRegressor

# Cargar tus datos
df = pd.read_csv("Datos_preprocesados.csv")

# Asegurar tipos correctos
df["zona"] = df["zona"].astype(str)

# Separar X e y
X = df[["zona", "habitaciones", "banos"]]
y = df["precio"]

# Definir modelo
model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=7,
    cat_features=["zona"],
    verbose=0
)

# Entrenar
model.fit(X, y)

# Guardar el modelo
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como model.pkl")