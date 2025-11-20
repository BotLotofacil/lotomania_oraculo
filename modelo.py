import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os
from dados import carregar_historico, carregar_estado, salvar_estado

ARQUIVO_MODELO = "lotomania_model.h5"

N_FEATURES = 100  # <-- CORRIGIDO (00 a 99)
JANELA = 50       # quantidade de concursos usados para prever o próximo

def criar_modelo():
    model = Sequential()
    model.add(LSTM(64, input_shape=(JANELA, N_FEATURES), return_sequences=False))
    model.add(Dense(N_FEATURES, activation="sigmoid"))
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model

def preparar_dados(df):
    X, y = [], []
    dezenas_cols = [f"{i:02d}" for i in range(N_FEATURES)]

    for i in range(len(df) - JANELA):
        bloco = df[dezenas_cols].iloc[i:i+JANELA].values
        proximo = df[dezenas_cols].iloc[i+JANELA].values
        X.append(bloco)
        y.append(proximo)

    return np.array(X), np.array(y)

def treinar_modelo():
    df = carregar_historico()
    estado = carregar_estado()

    ultimo_atual = int(df["concurso"].iloc[-1])

    if os.path.exists(ARQUIVO_MODELO):
        model = load_model(ARQUIVO_MODELO)
    else:
        model = criar_modelo()

    X, y = preparar_dados(df)

    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    model.save(ARQUIVO_MODELO)

    salvar_estado(ultimo_atual)

    return f"Treino concluído com {len(X)} exemplos."

def prever_proximo():
    df = carregar_historico()
    dezenas_cols = [f"{i:02d}" for i in range(N_FEATURES)]
    bloco = df[dezenas_cols].tail(JANELA).values.reshape(1, JANELA, N_FEATURES)

    model = load_model(ARQUIVO_MODELO)
    pred = model.predict(bloco)[0]

    return pred

def gerar_aposta():
    pred = prever_proximo()
    top20 = np.argsort(pred)[-20:]
    return sorted([int(n) for n in top20])

def gerar_aposta_espelho():
    pred = prever_proximo()
    bottom20 = np.argsort(pred)[:20]
    return sorted([int(n) for n in bottom20])

def gerar_errar_tudo():
    pred = prever_proximo()
    escolhas = np.argsort(pred)[:20]
    return sorted([int(n) for n in escolhas])
