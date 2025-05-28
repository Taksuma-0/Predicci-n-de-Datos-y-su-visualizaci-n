import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Configuración de la página
st.set_page_config(page_title="Resultados del MLP", layout="wide")
st.title("📊 Evaluación del Modelo MLP")

# Cargar datos preprocesados
@st.cache_data
def cargar_datos():
    # Cargar predicciones y valores reales desde archivo
    # Aquí se asume que ya guardaste los arreglos como .npz
    data = np.load("mlp_eval_resultados.npz")
    return data["y_test"], data["y_pred"], data.get("y_proba")

y_test, y_pred, y_proba = cargar_datos()

# Métricas básicas
st.subheader("🔍 Métricas de Evaluación")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.4f}")
col4.metric("F1-score", f"{f1_score(y_test, y_pred, average='macro'):.4f}")


import io

st.subheader("🧮 Matriz de Confusión")

# Generar figura compacta
fig, ax = plt.subplots(figsize=(4,3))  # Tamaño físico reducido
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax,
    cbar=False,
    annot_kws={"size": 10}
)

ax.set_xlabel("Predicción", fontsize=8)
ax.set_ylabel("Real", fontsize=8)
ax.tick_params(axis='both', labelsize=6)

# Guardar como imagen en buffer
buf = io.BytesIO()
fig.tight_layout(pad=0.5)
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)  # cerrar figura para no ocupar memoria

# Mostrar como imagen con tamaño controlado
st.image(buf, width=1000)  # puedes ajustar el ancho aquí


if y_proba is not None and len(np.unique(y_test)) == 2:
    st.subheader("📐 Curva ROC")

    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # Crear figura pequeña
    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax.set_xlabel("FPR", fontsize=8)
    ax.set_ylabel("TPR", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(loc="lower right", fontsize=6)

    # Guardar como imagen en buffer
    buf = io.BytesIO()
    fig.tight_layout(pad=0.5)
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Mostrar imagen reducida
    st.image(buf, width=1000)
