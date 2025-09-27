from knn_model import KNNModel
from mlp_model import MLPModel
from dtree_model import DecisionTreeModel

def main():
    # Cria e treina o modelo
    knn = KNNModel()
    knn.train("dataset/titato_train.csv")
    # Cria e treina o modelo
    mlp = MLPModel()
    mlp.train("dataset/titato_train.csv")
    # Cria e treina o modelo
    dtree = DecisionTreeModel()
    dtree.train("dataset/titato_train.csv")

    # Exemplo de entrada nova (mesmo número de colunas de treino!)
    novo_exemplo = [[1, -1, 0, 1, -1, 0, 1, -1, 1]]

    pred = knn.predict(novo_exemplo)
    print("Classe prevista (KNN):", pred[0])

    pred = mlp.predict(novo_exemplo)
    print("Classe prevista (MLP):", pred[0])

    pred = dtree.predict(novo_exemplo)
    print("Classe prevista (DecisionTree):", pred[0])

if __name__ == "__main__":
    main()