import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

ARQUIVO = "dados_letras.csv"

df = pd.read_csv(ARQUIVO)
print("Amostras por letra:")
print(df["letra"].value_counts().sort_index())
print()

X = df.drop(columns=["letra"])
y = df["letra"]

# Divisao 80% treino / 20% teste, mantendo a proporcao de cada letra
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
print(f"Acuracia no teste: {accuracy_score(y_teste, previsoes):.3f}\n")
print(classification_report(y_teste, previsoes))

# Matriz de confusao
letras = sorted(y.unique())
matriz = confusion_matrix(y_teste, previsoes, labels=letras)
ConfusionMatrixDisplay(matriz, display_labels=letras).plot(cmap="Blues")
plt.title("Matriz de confusao - Random Forest")
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
print("Matriz salva em matriz_confusao.png")

joblib.dump(modelo, "modelo_rf.joblib")
print("Modelo salvo em modelo_rf.joblib")

plt.show()