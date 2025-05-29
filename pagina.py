import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
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


st.set_page_config(page_title="Resultados del MLP", layout="wide")
st.title("📊 Evaluación del Modelo MLP")

# Cargar datos preprocesados del modelo
@st.cache_data
def cargar_datos():
    data = np.load("mlp_eval_resultados.npz")
    return data["y_test"], data["y_pred"], data.get("y_proba")

y_test, y_pred, y_proba = cargar_datos()

st.subheader("🔍 Métricas de Evaluación")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.4f}")
col4.metric("F1-score", f"{f1_score(y_test, y_pred, average='macro'):.4f}")


st.subheader("🧮 Matriz de Confusión")


fig, ax = plt.subplots(figsize=(4,3))  
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

## Guardamos en buffer, esto para que se vea en streamlit
buf = io.BytesIO()
fig.tight_layout(pad=0.5)
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)  


st.image(buf, width=1000) 


if y_proba is not None and len(np.unique(y_test)) == 2:
    st.subheader("📐 Curva ROC")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(2.5, 2.0))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax.set_xlabel("FPR", fontsize=8)
    ax.set_ylabel("TPR", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(loc="lower right", fontsize=6)

    buf = io.BytesIO()
    fig.tight_layout(pad=0.5)
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    st.image(buf, width=1000)
