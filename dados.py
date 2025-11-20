import pandas as pd
import json
import os

ARQUIVO_HISTORICO = "lotomania_historico_onehot.csv"
ARQUIVO_ESTADO = "estado.json"

def carregar_historico():
    df = pd.read_csv(ARQUIVO_HISTORICO, sep=";")
    return df

def salvar_estado(ultimo_concurso):
    estado = {"ultimo_concurso_treinado": int(ultimo_concurso)}
    with open(ARQUIVO_ESTADO, "w") as f:
        json.dump(estado, f)

def carregar_estado():
    if not os.path.exists(ARQUIVO_ESTADO):
        return {"ultimo_concurso_treinado": None}

    with open(ARQUIVO_ESTADO, "r") as f:
        return json.load(f)
