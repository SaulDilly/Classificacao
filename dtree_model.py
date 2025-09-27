import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=7,
            random_state=42,
            criterion="entropy"
        )

    def train(self, train_file):
        tictactoe = pd.read_csv(train_file)
        X_train = tictactoe.drop(columns=["state"])
        y_train = tictactoe["state"].values
        self.model.fit(X_train.values, y_train)

    def predict(self, new_data):
        # Retorna a predição do modelo treinado
        return self.model.predict(new_data)