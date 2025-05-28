import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
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

# 7. Entrenar Random Forest
#clf = RandomForestClassifier(class_weight="balanced")
# 3. Red neuronal multicapa con pipeline (normalización incluida)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(100, 50),activation="logistic", max_iter=1000,random_state=100))
])

# 4. Entrenar
pipe.fit(X_train, y_train)

# 5. Evaluar
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Reprobado", "Aprobado"])
disp.plot(cmap="Purples")
plt.title("Matriz de Confusión - MLPClassifier")
plt.grid(False)
plt.show()
# 7. Guardar modelo
joblib.dump(pipe, "mlp_aprobacion.joblib")


# Obtener probabilidades de clase positiva
y_scores = pipe.predict_proba(X_test)[:, 1]  # solo la columna para clase 1

# Calcular fpr, tpr
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calcular área bajo la curva
auc = roc_auc_score(y_test, y_scores)
print(f"AUC ROC: {auc:.4f}")

# Graficar curva ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"MLPClassifier (AUC = {auc:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("Tasa de falsos positivos (FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("Curva ROC - MLPClassifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()