import pandas as pd
from xgboost import XGBClassifier

class XGBModel:
    def __init__(self):
        # Parâmetros obtidos através de busca sistemática
        # Max_depth -> profundidade de cada árvore
        # N_estimators -> número de árvores
        # subsample -> porcentagem dos dados que serão usados em cada árvore
        self.model = XGBClassifier(
            learning_rate=0.1,
            max_depth=3,
            n_estimators=400,
            subsample=0.6,
            random_state=42
        )
    def train(self, train_file):
        data_train = pd.read_csv(train_file)
        X_train = data_train.drop(columns=["state"])
        y_train = data_train["state"].values
        self.model.fit(X_train.values, y_train)

    def predict(self, new_data):
        return self.model.predict(new_data)
