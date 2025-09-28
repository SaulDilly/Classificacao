from knn_model import KNNModel
from mlp_model import MLPModel
from dtree_model import DecisionTreeModel
import random


def verificaTemJogo(ttt):
    if(verificaGanhador(ttt) == -1):
        return 0
    if(verificaPosFimDeJogo(ttt) == -1):
        return 0
    return -1


def verificaPosFimDeJogo(ttt):
    # Pos. Fim de Jogo nas linhas
    if (ttt[0] == 1 and ttt[1] == 1 and ttt[2] == -1): return 2
    if (ttt[0] == -1 and ttt[1] == 1 and ttt[2] == 1): return 2
    if (ttt[3] == 1 and ttt[4] == 1 and ttt[5] == -1): return 2
    if (ttt[3] == -1 and ttt[4] == 1 and ttt[5] == 1): return 2
    if (ttt[6] == 1 and ttt[7] == 1 and ttt[8] == -1): return 2
    if (ttt[6] == -1 and ttt[7] == 1 and ttt[8] == 1): return 2
    if (ttt[0] == 0 and ttt[1] == 0 and ttt[2] == -1): return 2
    if (ttt[0] == -1 and ttt[1] == 0 and ttt[2] == 0): return 2
    if (ttt[3] == 0 and ttt[4] == 0 and ttt[5] == -1): return 2
    if (ttt[3] == -1 and ttt[4] == 0 and ttt[5] == 0): return 2
    if (ttt[6] == 0 and ttt[7] == 0 and ttt[8] == -1): return 2
    if (ttt[6] == -1 and ttt[7] == 0 and ttt[8] == 0): return 2

    # Pos. Fim de Jogo nas colunas
    if (ttt[0] == 1 and ttt[3] == 1 and ttt[6] == -1): return 2
    if (ttt[0] == -1 and ttt[3] == 1 and ttt[6] == 1): return 2
    if (ttt[1] == 1 and ttt[4] == 1 and ttt[7] == -1): return 2
    if (ttt[1] == -1 and ttt[4] == 1 and ttt[7] == 1): return 2
    if (ttt[2] == 1 and ttt[5] == 1 and ttt[8] == -1): return 2
    if (ttt[2] == -1 and ttt[5] == 1 and ttt[8] == 1): return 2
    if (ttt[0] == 0 and ttt[3] == 0 and ttt[6] == -1): return 2
    if (ttt[0] == -1 and ttt[3] == 0 and ttt[6] == 0): return 2
    if (ttt[1] == 0 and ttt[4] == 0 and ttt[7] == -1): return 2
    if (ttt[1] == -1 and ttt[4] == 0 and ttt[7] == 0): return 2
    if (ttt[2] == 0 and ttt[5] == 0 and ttt[8] == -1): return 2
    if (ttt[2] == -1 and ttt[5] == 0 and ttt[8] == 0): return 2

    # Pos. Fim de Jogo nas diagonais
    if (ttt[0] == 1 and ttt[4] == 1 and ttt[8] == -1): return 2
    if (ttt[0] == -1 and ttt[4] == 1 and ttt[8] == 1): return 2
    if (ttt[6] == 1 and ttt[4] == 1 and ttt[2] == -1): return 2
    if (ttt[6] == -1 and ttt[4] == 1 and ttt[2] == 1): return 2
    if (ttt[0] == 0 and ttt[4] == 0 and ttt[8] == -1): return 2
    if (ttt[0] == -1 and ttt[4] == 0 and ttt[8] == 0): return 2
    if (ttt[6] == 0 and ttt[4] == 0 and ttt[2] == -1): return 2
    if (ttt[6] == -1 and ttt[4] == 0 and ttt[2] == 0): return 2

    return -1



def verificaGanhador(ttt):
    # Verifica se 1 ganhou nas linhas
    if (ttt[0] == 1 and ttt[1] == 1 and ttt[2] == 1): return 1
    if (ttt[3] == 1 and ttt[4] == 1 and ttt[5] == 1): return 1
    if (ttt[6] == 1 and ttt[7] == 1 and ttt[8] == 1): return 1

    # Verifica se 1 ganhou nas colunas
    if (ttt[0] == 1 and ttt[3] == 1 and ttt[6] == 1): return 1
    if (ttt[1] == 1 and ttt[4] == 1 and ttt[7] == 1): return 1
    if (ttt[2] == 1 and ttt[5] == 1 and ttt[8] == 1): return 1

    # Verifica se 1 ganhou nas diagonais
    if (ttt[0] == 1 and ttt[4] == 1 and ttt[8] == 1): return 1
    if (ttt[6] == 1 and ttt[4] == 1 and ttt[2] == 1): return 1

    # Verifica se 0 ganhou nas linhas
    if (ttt[0] == 0 and ttt[1] == 0 and ttt[2] == 0): return 1
    if (ttt[3] == 0 and ttt[4] == 0 and ttt[5] == 0): return 1
    if (ttt[6] == 0 and ttt[7] == 0 and ttt[8] == 0): return 1

    # Verifica se 0 ganhou nas colunas
    if (ttt[0] == 0 and ttt[3] == 0 and ttt[6] == 0): return 1
    if (ttt[1] == 0 and ttt[4] == 0 and ttt[7] == 0): return 1
    if (ttt[2] == 0 and ttt[5] == 0 and ttt[8] == 0): return 1

    # Verifica se 0 ganhou nas diagonais
    if (ttt[0] == 0 and ttt[4] == 0 and ttt[8] == 0): return 1
    if (ttt[6] == 0 and ttt[4] == 0 and ttt[2] == 0): return 1

    return -1

def estadoAtual(ttt):
    if (verificaGanhador(ttt) == 1):
        return 1
    if (verificaPosFimDeJogo(ttt) == 2):
        return 2
    if(verificaTemJogo(ttt) == 0):
        return 0
    return -1

def imprimeGuia():
    print("Posições do tabuleiro:")
    print("\n")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")
    print("\n")
    print("Escolha um número de 0 a 8 para jogar!\n")

def imprimeTabuleiro(ttt):
    print("\n")
    print(f" {ttt[0]} | {ttt[1]} | {ttt[2]} ")
    print("---+---+---")
    print(f" {ttt[3]} | {ttt[4]} | {ttt[5]} ")
    print("---+---+---")
    print(f" {ttt[6]} | {ttt[7]} | {ttt[8]} ")
    print("\n")


def simulaJogo():

    knn = KNNModel()
    knn.train("dataset/titato_train.csv")

    jogador = 1
    maquina = 0
    titato = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    acertos = 0
    erros = 0
    total_preds = 0
    partida = 1
    turno = jogador

    imprimeGuia()

    while True:
        print("Partida: ", partida)
        imprimeTabuleiro(titato)
        entrada_modelo = [titato.copy()]

        pred = knn.predict(entrada_modelo)
        estado_atual = estadoAtual(titato)

        if estado_atual == pred[0]:
            acertos += 1
        else:
            erros += 1
        total_preds += 1

        # Exibe informações solicitadas
        print(f"IA analisando: KNN")
        print(f"Predição da IA (estado do jogo): {pred[0]}")
        print(f"Estado real do jogo: {estado_atual}")
        print(f"Acertos: {acertos} | Erros: {erros}")

        if turno == jogador:
            while True:
                # Vez do jogador
                jogada = input("Sua Jogada (0-8): ")
                pos = int(jogada)
                if (titato[pos] == -1):
                    titato[pos] = jogador
                    turno = maquina
                    break

        else:
            # Vez da máquina
            while True:
                pos = random.randint(0,8)
                if(titato[pos] == -1):
                    titato[pos] = maquina
                    turno = jogador
                    break

        if estado_atual == 1:
            partida += 1
            break


    print("\n=== Estatísticas finais ===")
    print(f"Total de predições realizadas: {total_preds}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")





if __name__ == "__main__":
    simulaJogo()


























