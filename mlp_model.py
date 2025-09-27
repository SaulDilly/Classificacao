import pandas as pd
from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self):
        # Parâmetros obtidos através de busca sistemática
        self.model = MLPClassifier(
            solver='adam',
            hidden_layer_sizes=(20,),
            learning_rate_init=0.001,
            momentum=0.5,
            random_state=42,
            verbose=False
        )

    def train(self, train_file):
        data_train = pd.read_csv(train_file)
        X_train = data_train.drop(columns=["state"])
        y_train = data_train["state"].values
        self.model.fit(X_train.values, y_train)

    def predict(self, new_data):
        # Retorna a predição do modelo treinado
        return self.model.predict(new_data)
