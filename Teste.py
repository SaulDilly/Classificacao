import pandas as pd
import numpy as np
# 1. MUDANÇA: Importar o KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Carregar os dados (sem alterações)
try:
    train = pd.read_csv("dataset/titato_train.csv")
    test = pd.read_csv("dataset/titato_test.csv")
except FileNotFoundError as e:
    print(f"❌ Erro: Arquivo não encontrado! Verifique o caminho: {e.filename}")
    exit()


# A última coluna é a classe
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

# =========================
# TREINAR O CLASSIFICADOR
# =========================
# 2. MUDANÇA: Substituir o classificador anterior pelo KNeighborsClassifier
# O parâmetro 'k' é chamado de 'n_neighbors' no scikit-learn
clf = KNeighborsClassifier(
    n_neighbors=3
)

print(" Neighbors) com k=3...")
# Para o KNN, o "fit" é muito rápido, pois ele apenas armazena os dados.
clf.fit(X_train, y_train)
print("✅ Modelo 'treinado' (dados armazenados).")

# =========================
# TESTAR O CLASSIFICADOR
# =========================
y_pred = clf.predict(X_test)

# Matriz de confusão (sem alterações)
# Apenas ajustamos o título e a cor para clareza
print("\nGerando a Matriz de Confusão...")
classes_unicas = np.unique(y_train)
cm = confusion_matrix(y_test, y_pred, labels=classes_unicas)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_unicas)
disp.plot(cmap='Purples')
plt.title("Matriz de Confusão - KNN (k=3)")
plt.show()

# Métricas (sem alterações)
print("\n--- 📊 Métricas de Avaliação ---")
print(f"Acurácia (sklearn): {accuracy_score(y_test, y_pred):.4f}")
print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred))

# Informações do modelo
print("\n--- Informações do Modelo ---")
# 3. MUDANÇA: Mostrar os parâmetros usados pelo KNN
print("Classes:", clf.classes_)
print(f"Número de vizinhos (k): {clf.n_neighbors}")
print("Parâmetros:", clf.get_params())