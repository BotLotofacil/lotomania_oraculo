import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

from dados import carregar_historico, carregar_estado, salvar_estado

ARQUIVO_MODELO = "lotomania_model.npz"


def _colunas_dezenas():
    # colunas 0..99 da planilha one-hot
    return [str(i) for i in range(100)]


def criar_modelo(n_features: int = 100):
    model = Sequential()
    model.add(LSTM(64, input_shape=(50, n_features), return_sequences=False))
    model.add(Dense(n_features, activation="sigmoid"))
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
    return model


def preparar_dados(df):
    dezenas_cols = _colunas_dezenas()
    X, y = [], []

    # janelas de 50 concursos -> prevê o próximo
    for i in range(len(df) - 50):
        bloco = df[dezenas_cols].iloc[i:i + 50].values
        proximo = df[dezenas_cols].iloc[i + 50].values
        X.append(bloco)
        y.append(proximo)

    return np.array(X), np.array(y)


def treinar_modelo(epochs: int = 15):
    df = carregar_historico()
    if len(df) < 51:
        return "Poucos concursos para treinar (mínimo 51 linhas)."

    estado = carregar_estado()
    ultimo_atual = int(df["concurso"].iloc[-1])

    if os.path.exists(ARQUIVO_MODELO):
        model = load_model(ARQUIVO_MODELO)
    else:
        model = criar_modelo()

    X, y = preparar_dados(df)

    if len(X) == 0:
        return "Dataset vazio após preparação."

    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    model.save(ARQUIVO_MODELO)

    salvar_estado(ultimo_atual)
    return f"Treino concluído. Último concurso treinado: {ultimo_atual}"


def prever_proximo():
    """Retorna vetor de probabilidades normalizado (100 dezenas)."""
    df = carregar_historico()
    dezenas_cols = _colunas_dezenas()
    if len(df) < 50:
        raise RuntimeError("Histórico insuficiente para prever (mínimo 50 linhas).")

    bloco = df[dezenas_cols].tail(50).values.reshape(1, 50, len(dezenas_cols))

    model = load_model(ARQUIVO_MODELO)
    pred = model.predict(bloco, verbose=0)[0]

    # evita zeros e normaliza
    pred = np.clip(pred, 1e-6, None)
    pred = pred / pred.sum()
    return pred


def _frequencias(df, janela: int = 200):
    """Frequência de cada dezena na janela recente."""
    dezenas_cols = _colunas_dezenas()
    janela = min(janela, len(df))
    recorte = df.tail(janela)
    return recorte[dezenas_cols].sum().to_numpy()


def _ultimo_resultado(df):
    """Lista de dezenas sorteadas no último concurso."""
    dezenas_cols = _colunas_dezenas()
    linha = df[dezenas_cols].tail(1).iloc[0].to_numpy().astype(int)
    return [i for i, v in enumerate(linha) if v == 1]


def _atrasos(df):
    """Quantos concursos cada dezena está sem sair (atraso)."""
    dezenas_cols = _colunas_dezenas()
    n = len(df)
    atrasos = np.full(100, n, dtype=int)
    arr = df[dezenas_cols].to_numpy()

    for dez in range(100):
        col = arr[:, dez]
        idxs = np.where(col == 1)[0]
        if len(idxs) > 0:
            atrasos[dez] = n - 1 - idxs[-1]

    return atrasos


def gerar_apostas_oraculo_supremo():
    """
    Retorna:
        apostas: lista com 6 apostas (cada uma 50 dezenas)
        espelhos: lista com 6 espelhos (complemento das 50 dezenas)
    """
    df = carregar_historico()
    dezenas_cols = _colunas_dezenas()
    if len(df) < 50:
        raise RuntimeError("Histórico insuficiente para gerar apostas (mínimo 50 linhas).")

    pred = prever_proximo()                          # probabilidade por dezena
    freq = _frequencias(df, janela=200)              # dezenas quentes/frias
    atrasos = _atrasos(df)                           # atraso em concursos
    ult = _ultimo_resultado(df)                      # último resultado

    rng = np.random.default_rng()

    # Pequeno ruído para evitar apostas idênticas
    pred_ruido = pred + rng.normal(0, 0.01, size=pred.shape)
    pred_ruido = np.clip(pred_ruido, 1e-6, None)
    pred_ruido = pred_ruido / pred_ruido.sum()

    # ------------ APOSTA 1 – REPETIÇÃO ------------
    # Dá peso forte para as dezenas do último concurso + probabilidade do modelo
    candidatos_rep = sorted(ult, key=lambda d: pred[d], reverse=True)
    aposta1 = candidatos_rep[:30]  # até 30 repetidas

    restantes1 = [int(d) for d in np.argsort(pred)[::-1] if d not in aposta1]
    aposta1 += restantes1[:(50 - len(aposta1))]
    aposta1 = sorted(aposta1)

    # ------------ APOSTA 2 – CICLOS ------------
    # Combina atraso + probabilidade (dezena que está "no ponto" de voltar)
    score_ciclo = pred * (1.0 + atrasos / (len(df) + 1.0))
    idx_ciclos = np.argsort(score_ciclo)[-50:]
    aposta2 = sorted(idx_ciclos.tolist())

    # ------------ APOSTA 3 – PROBABILÍSTICA PURA ------------
    aposta3 = sorted(rng.choice(100, size=50, replace=False, p=pred_ruido).tolist())

    # ------------ APOSTA 4 – HÍBRIDA (TOP + MÉDIAS) ------------
    ordem_pred = np.argsort(pred)
    top = ordem_pred[-30:]            # mais prováveis
    medios = ordem_pred[20:80]        # faixa intermediária

    top_list = list(top)
    # escolher dezenas da faixa média sem repetir
    restantes4 = [int(d) for d in medios if d not in top_list]
    qtd_top = 25
    qtd_medios = 50 - qtd_top

    if len(restantes4) >= qtd_medios:
        extra4 = rng.choice(restantes4, size=qtd_medios, replace=False).tolist()
    else:
        extra4 = restantes4

    aposta4 = sorted(top_list[-qtd_top:] + extra4)

    # ------------ APOSTA 5 – DEZENAS QUENTES ------------
    idx_quentes = np.argsort(freq)[-50:]
    aposta5 = sorted(idx_quentes.tolist())

    # ------------ APOSTA 6 – DEZENAS FRIAS ------------
    idx_frias = np.argsort(freq)[:50]
    aposta6 = sorted(idx_frias.tolist())

    apostas = [aposta1, aposta2, aposta3, aposta4, aposta5, aposta6]

    # ------------ ESPELHOS ------------
    universo = set(range(100))
    espelhos = [sorted(universo - set(ap)) for ap in apostas]

    return apostas, espelhos
