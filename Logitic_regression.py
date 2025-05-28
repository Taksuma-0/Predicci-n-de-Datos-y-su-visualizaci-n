import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("data/student-por.csv")

# Crear variable binaria: 1 = aprueba, 0 = reprueba
df["passed"] = (df["G3"] >= 10).astype(int) ## Target para posterior predicción

# Eliminar G1 y G2
df = df.drop(columns=["G1", "G2", "G3"])

#  One-hot encoding para variables categóricas
df = pd.get_dummies(df, drop_first=True)

## Obtenemos set de datos de features y target, separandolos
X = df.drop(columns=["passed"])
y = df["passed"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=100)

# 3. Aplicar SMOTE para balancear entrenamiento
smote = SMOTE(random_state=100)
X_res, y_res = smote.fit_resample(X_train, y_train)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(100, 50),activation="logistic", max_iter=1000,random_state=100))
])

# 4. Entrenar
pipe.fit(X_train, y_train)

# 7. Guardar modelo
joblib.dump(pipe, "mlp_aprobacion.joblib")

# Cargar modelo entrenado
modelo = joblib.load("mlp_aprobacion.joblib")

# Separar test (o usa directamente tu X_test, y_test)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predicción
y_pred = modelo.predict(X_test)

# Probabilidades (solo si el modelo soporta)
try:
    y_proba = modelo.predict_proba(X_test)
except:
    y_proba = None

# Guardar resultados para Streamlit
np.savez("mlp_eval_resultados.npz", y_test=y_test, y_pred=y_pred, y_proba=y_proba)

print("✅ Evaluación guardada en 'mlp_eval_resultados.npz'")
