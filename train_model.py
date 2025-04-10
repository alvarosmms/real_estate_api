import pandas as pd
import pickle
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Función para evaluar un modelo en el conjunto de test
def evaluate_model(model_pipeline, X_test, y_test):
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

# Cargar tus datos
df = pd.read_csv("Datos_preprocesados.csv")

# Asegurar tipos correctos
df["zona"] = df["zona"].astype(str)

# Separar X e y
X = df[["zona", "habitaciones", "banos", "tipovivienda", "metros"]]
y = df["precio"]

# División en conjuntos de entrenamiento y test (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo
model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=7,
    cat_features=["zona", "tipovivienda"],
    eval_metric="RMSE",
    random_seed=42,
    verbose=0
)

# Entrenar
model.fit(X, y)

results = {}
model_pipelines = {}

test_mse, test_mae, test_r2 = evaluate_model(model, X_test, y_test)

print(f"Modelo: Catboost")
print(f"  MSE (Test): {test_mse:.2f}")
print(f"  MAE (Test): {test_mae:.2f}")
print(f"  R² (Test): {test_r2:.2f}\n")

# Guardar el modelo
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "mae": test_mae}, f)

print("✅ Modelo entrenado y guardado como model.pkl")