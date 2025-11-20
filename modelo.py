import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os
from dados import carregar_historico, carregar_estado, salvar_estado

ARQUIVO_MODELO = "lotomania_model.npz"

def criar_modelo(n_features=74):
    model = Sequential()
    model.add(LSTM(64, input_shape=(50, n_features), return_sequences=False))
    model.add(Dense(74, activation="sigmoid"))
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model

def preparar_dados(df):
    X, y = [], []
    dezenas_cols = [str(i) for i in range(74)]

    for i in range(len(df) - 50):
        bloco = df[dezenas_cols].iloc[i:i+50].values
        proximo = df[dezenas_cols].iloc[i+50].values
        X.append(bloco)
        y.append(proximo)

    return np.array(X), np.array(y)

def treinar_modelo():
    df = carregar_historico()
    estado = carregar_estado()

    ultimo_salvo = estado.get("ultimo_concurso_treinado")
    ultimo_atual = int(df["concurso"].iloc[-1])

    # Carregar modelo existente
    if os.path.exists(ARQUIVO_MODELO):
        model = load_model(ARQUIVO_MODELO)
    else:
        model = criar_modelo()

    # Preparar dataset completo
    X, y = preparar_dados(df)

    # Treinar (não zera)
    model.fit(X, y, epochs=15, batch_size=16, verbose=0)

    # Salvar modelo
    model.save(ARQUIVO_MODELO)

    # atualizar estado
    salvar_estado(ultimo_atual)

    return f"Treino concluído. Último concurso treinado: {ultimo_atual}"

def prever_proximo():
    from tensorflow.keras.models import load_model

    df = carregar_historico()
    dezenas_cols = [str(i) for i in range(74)]
    bloco = df[dezenas_cols].tail(50).values.reshape(1, 50, 74)

    model = load_model(ARQUIVO_MODELO)
    pred = model.predict(bloco)[0]

    return pred  # vetor com 74 probabilidades

def gerar_aposta():
    pred = prever_proximo()
    top20 = np.argsort(pred)[-20:]     # 20 mais prováveis
    top20_sorted = sorted(top20)
    return [int(n) for n in top20_sorted]

def gerar_aposta_espelho():
    pred = prever_proximo()
    bottom20 = np.argsort(pred)[:20]   # 20 menos prováveis
    bottom20_sorted = sorted(bottom20)
    return [int(n) for n in bottom20_sorted]

def gerar_errar_tudo():
    """Seleciona as 20 piores dezenas possíveis."""
    pred = prever_proximo()
    escolhas = np.argsort(pred)[:20]
    escolhas_sorted = sorted(escolhas)
    return [int(n) for n in escolhas_sorted]
