import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self):
    # k=3 -> Melhor resultado, testando em um intervalo (2, 30)
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, train_file):
        tictactoe = pd.read_csv(train_file)
        X_train = tictactoe.drop(columns=["state"])
        y_train = tictactoe["state"].values
        self.model.fit(X_train.values, y_train)

    def predict(self, new_data):
        # Retorna a predição do modelo treinado
        return self.model.predict(new_data)