# ORACULO.py – Oráculo Lotomania – Modo C (Híbrido CNN + MLP)

import json
import time
import os
import csv
import logging
from typing import List, Set

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ----------------------------------------------------
# Caminhos de arquivos principais
# ----------------------------------------------------
HISTORY_PATH = "lotomania_historico_onehot.csv"   # histórico one-hot (00–99)
MODEL_PATH = "lotomania_model.npz"                # pesos da rede neural

# Arquivo para guardar o último lote gerado (para o /confirmar)
ULTIMA_GERACAO_PATH = "ultima_geracao_oraculo.json"

# Arquivo de telemetria de desempenho
DESEMPENHO_PATH = "desempenho_oraculo.csv"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# REDE NEURAL HÍBRIDA – CNN 1D + MLP
# ----------------------------------------------------


class HybridCNNMLP:
    """
    Modelo híbrido:
      - Branch 1: sequência binária (0/1) dos últimos seq_len concursos para cada dezena (CNN 1D)
      - Branch 2: features manuais (freq 10/20/50 + gap) -> MLP
      - Saída: probabilidade da dezena sair no próximo concurso (sigmoid)
    """

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        conv_channels: int = 8,
        kernel_size: int = 5,
        hidden_dim: int = 32,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.lr = lr

        rng = np.random.default_rng(seed)

        # Convolução 1D: kernel_size x conv_channels
        self.Wc = rng.normal(0, 0.1, size=(kernel_size, conv_channels)).astype(
            np.float32
        )
        self.bc = np.zeros((conv_channels,), dtype=np.float32)

        # MLP (entrada = features CNN + features manuais)
        in_dim = conv_channels + feat_dim
        self.W1 = rng.normal(0, 0.1, size=(in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, size=(hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros((1,), dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _forward_batch(self, X_ts: np.ndarray, X_feat: np.ndarray):
        """
        X_ts   : (B, seq_len)       – sequência 0/1
        X_feat : (B, feat_dim)      – features manuais já escaladas
        """
        B, L = X_ts.shape
        K = self.kernel_size

        # Cria janelas deslizantes para convolução: (B, L-K+1, K)
        windows = sliding_window_view(X_ts, K, axis=1)
        B, Lw, K2 = windows.shape
        assert K2 == K

        # Conv 1D: (B, Lw, K) @ (K, C) + bc -> (B, Lw, C)
        conv_out = windows @ self.Wc + self.bc  # broadcast em C
        conv_act = self._relu(conv_out)

        # Global Average Pooling sobre o eixo temporal (Lw)
        conv_feat = conv_act.mean(axis=1)  # (B, C)

        # Concatena CNN + features manuais
        concat = np.concatenate([conv_feat, X_feat], axis=1)  # (B, C + F)

        # MLP
        z1 = concat @ self.W1 + self.b1
        a1 = self._tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self._sigmoid(z2)  # (B, 1)

        cache = (windows, conv_out, conv_act, conv_feat, concat, a1, y_hat)
        return y_hat, cache

    def fit(
        self,
        X_ts: np.ndarray,
        X_feat: np.ndarray,
        y: np.ndarray,
        epochs: int = 5,
        batch_size: int = 512,
    ):
        """
        Treino com mini-batches.
        Se já houver pesos, continua o aprendizado (não reseta).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        N = X_ts.shape[0]

        for _ in range(epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            X_ts_sh = X_ts[idx]
            X_feat_sh = X_feat[idx]
            y_sh = y[idx]

            for start in range(0, N, batch_size):
                end = start + batch_size
                Xb_ts = X_ts_sh[start:end]
                if Xb_ts.shape[0] == 0:
                    continue

                Xb_feat = X_feat_sh[start:end]
                yb = y_sh[start:end]
                B = Xb_ts.shape[0]

                y_pred, cache = self._forward_batch(Xb_ts, Xb_feat)
                (
                    windows,
                    conv_out,
                    conv_act,
                    conv_feat,
                    concat,
                    a1,
                    y_hat,
                ) = cache

                # Gradiente BCE (mesmo esquema do MLP simples)
                m = B
                dz2 = (y_hat - yb) / m  # (B, 1)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T  # (B, hidden_dim)
                dz1 = da1 * (1.0 - a1 ** 2)  # derivada da tanh
                dW1 = concat.T @ dz1
                db1 = dz1.sum(axis=0)

                dconcat = dz1 @ self.W1.T  # (B, conv_channels + feat_dim)

                C = self.conv_channels
                dconv_feat = dconcat[:, :C]  # gradiente na saída do branch CNN
                # dX_feat = dconcat[:, C:]  # gradiente nas features manuais (não usado externamente)

                # Backprop no pooling:
                # conv_feat[b, c] = mean_t conv_act[b, t, c]
                B2, Lw, C2 = conv_act.shape
                assert B2 == B and C2 == C

                dconv_act = np.zeros_like(conv_act)
                dconv_act[:] = (dconv_feat[:, None, :] / float(Lw))

                # Backprop ReLU
                dconv_out = dconv_act * (conv_out > 0)

                # conv_out[b, t, c] = sum_k windows[b, t, k] * Wc[k, c] + bc[c]
                wl = windows.reshape(-1, self.kernel_size)  # (B*Lw, K)
                dl = dconv_out.reshape(-1, C)              # (B*Lw, C)

                dWc = wl.T @ dl                             # (K, C)
                dbc = dconv_out.sum(axis=(0, 1))            # (C,)

                # Atualização dos pesos
                lr = self.lr
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.Wc -= lr * dWc
                self.bc -= lr * dbc

    def predict_proba(self, X_ts: np.ndarray, X_feat: np.ndarray) -> np.ndarray:
        y_hat, _ = self._forward_batch(X_ts, X_feat)
        return y_hat.ravel()


class ModelWrapper:
    """
    Wrapper para salvar/carregar o modelo + scaler + metadados.
    """

    def __init__(
        self,
        net: HybridCNNMLP,
        mean_feat: np.ndarray,
        std_feat: np.ndarray,
        seq_len: int,
    ):
        self.net = net
        self.mean_feat = mean_feat
        self.std_feat = std_feat
        self.seq_len = seq_len


_model_cache: ModelWrapper | None = None  # cache na memória

# ----------------------------------------------------
# LEITURA DO HISTÓRICO
# ----------------------------------------------------


def load_history(path: str) -> List[Set[int]]:
    """
    Lê o CSV one-hot no formato:
    concurso;data;00;01;...;99
    ou
    concurso;data;0;1;...;99

    Retorna lista de conjuntos de dezenas sorteadas por concurso.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de histórico não encontrado: {path}")

    history: List[Set[int]] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)

        # primeiros 2 campos = concurso, data; o resto são dezenas
        dez_cols = header[2:]
        dez_map = [int(col) for col in dez_cols]  # converte "00" -> 0, "1" -> 1 etc.

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue

            dezenas = set()
            for val, dez in zip(row[2:], dez_map):
                if val.strip() == "1":
                    dezenas.add(dez)

            history.append(dezenas)

    return history


# ----------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------


def compute_features_for_dozen(
    history: List[Set[int]],
    idx: int,
    dezena: int,
    max_window: int = 50,
):
    """
    Calcula features da dezena em um ponto da linha do tempo:
    - freq nos últimos 10, 20, 50
    - gap (concursos sem sair)
    """
    start = max(0, idx - max_window + 1)
    janela = history[start: idx + 1]

    def freq_ultimos(n: int) -> int:
        sub = janela[-n:] if len(janela) >= n else janela
        return sum(1 for conc in sub if dezena in conc)

    f10 = freq_ultimos(10)
    f20 = freq_ultimos(20)
    f50 = freq_ultimos(50)

    # gap
    gap = 0
    encontrado = False
    for k in range(len(janela) - 1, -1, -1):
        if dezena in janela[k]:
            encontrado = True
            break
        gap += 1
    if not encontrado:
        gap = max_window + 1

    return [float(f10), float(f20), float(f50), float(gap)]


def compute_ts_for_dozen(
    history: List[Set[int]],
    idx: int,
    dezena: int,
    seq_len: int,
):
    """
    Retorna vetor 0/1 de presença da dezena nos últimos seq_len concursos
    (da posição idx-seq_len+1 até idx).
    """
    ts = []
    start = idx - seq_len + 1
    for t in range(start, idx + 1):
        if t < 0:
            ts.append(0.0)
        else:
            ts.append(1.0 if dezena in history[t] else 0.0)
    return ts


def build_dataset_hybrid(history: List[Set[int]], seq_len: int):
    """
    Gera X_ts, X_feat, y para treino híbrido:
      - X_ts   : sequência 0/1 (últimos seq_len concursos)
      - X_feat : features manuais (freq 10/20/50 + gap)
      - y      : 1 se a dezena saiu no concurso seguinte, 0 se não
    """
    X_ts_list = []
    X_feat_list = []
    y_list = []

    n = len(history)
    if n < seq_len + 1:
        raise ValueError(
            f"Histórico insuficiente para seq_len={seq_len}. Tamanho histórico: {n}"
        )

    # idx representa o concurso base. Target é history[idx + 1].
    for idx in range(seq_len - 1, n - 1):
        proximo = history[idx + 1]
        for dezena in range(100):  # 00 a 99
            ts = compute_ts_for_dozen(history, idx, dezena, seq_len)
            feats = compute_features_for_dozen(history, idx, dezena)
            X_ts_list.append(ts)
            X_feat_list.append(feats)
            y_list.append(1 if dezena in proximo else 0)

    X_ts = np.array(X_ts_list, dtype=np.float32)
    X_feat = np.array(X_feat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X_ts, X_feat, y


def compute_scaler(X_feat: np.ndarray):
    """
    Scaler apenas para as features manuais (não para a sequência 0/1).
    """
    mean_ = X_feat.mean(axis=0)
    std_ = X_feat.std(axis=0)
    std_[std_ == 0] = 1.0
    return mean_, std_


# ----------------------------------------------------
# SALVAR / CARREGAR MODELO
# ----------------------------------------------------


def save_model(wrapper: ModelWrapper, path: str = MODEL_PATH):
    net = wrapper.net
    np.savez(
        path,
        Wc=net.Wc,
        bc=net.bc,
        W1=net.W1,
        b1=net.b1,
        W2=net.W2,
        b2=net.b2,
        mean_feat=wrapper.mean_feat,
        std_feat=wrapper.std_feat,
        seq_len=np.array([wrapper.seq_len], dtype=np.int32),
        feat_dim=np.array([net.feat_dim], dtype=np.int32),
        conv_channels=np.array([net.conv_channels], dtype=np.int32),
        kernel_size=np.array([net.kernel_size], dtype=np.int32),
        hidden_dim=np.array([net.hidden_dim], dtype=np.int32),
        lr=np.array([net.lr], dtype=np.float32),
    )


def load_model_local(path: str = MODEL_PATH) -> ModelWrapper:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(path):
        raise FileNotFoundError("Modelo ainda não treinado. Use /treinar.")

    data = np.load(path)

    # Verifica se é modelo antigo (sem Wc) e força recriação
    if "Wc" not in data.files:
        raise RuntimeError(
            "Modelo salvo é de versão antiga (sem CNN). "
            "Apague o arquivo 'lotomania_model.npz' e rode /treinar novamente."
        )

    Wc = data["Wc"]
    bc = data["bc"]
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    mean_feat = data["mean_feat"]
    std_feat = data["std_feat"]
    seq_len = int(data["seq_len"][0])
    feat_dim = int(data["feat_dim"][0])
    conv_channels = int(data["conv_channels"][0])
    kernel_size = int(data["kernel_size"][0])
    hidden_dim = int(data["hidden_dim"][0])
    lr = float(data["lr"][0])

    net = HybridCNNMLP(
        seq_len=seq_len,
        feat_dim=feat_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        lr=lr,
    )
    net.Wc = Wc
    net.bc = bc
    net.W1 = W1
    net.b1 = b1
    net.W2 = W2
    net.b2 = b2

    wrapper = ModelWrapper(net, mean_feat, std_feat, seq_len)
    _model_cache = wrapper
    return wrapper


# ----------------------------------------------------
# GERAÇÃO DE PROBABILIDADES E APOSTAS
# ----------------------------------------------------


def gerar_probabilidades_para_proximo(
    history: List[Set[int]], model: ModelWrapper
) -> np.ndarray:
    """
    Gera vetor de probabilidades (00–99) para o PRÓXIMO concurso,
    usando o último ponto da linha do tempo do histórico.
    """
    if len(history) < 2:
        raise ValueError("Histórico insuficiente para prever o próximo concurso.")

    idx = len(history) - 1  # último concurso conhecido
    seq_len = model.seq_len
    net = model.net

    ts_list = []
    feat_list = []

    for dezena in range(100):
        ts = compute_ts_for_dozen(history, idx, dezena, seq_len)
        feats = compute_features_for_dozen(history, idx, dezena)
        ts_list.append(ts)
        feat_list.append(feats)

    X_ts = np.array(ts_list, dtype=np.float32)
    X_feat = np.array(feat_list, dtype=np.float32)

    # escala apenas as features manuais
    X_feat_scaled = (X_feat - model.mean_feat) / model.std_feat

    probs = net.predict_proba(X_ts, X_feat_scaled)
    return probs  # shape (100,)


def gerar_apostas_e_espelhos(history: List[Set[int]], model: ModelWrapper):
    """
    Versão simples – não usada diretamente no Oráculo Supremo, mas mantida se precisar.
    """
    probs = gerar_probabilidades_para_proximo(history, model)
    idx_sorted_desc = np.argsort(-probs).tolist()  # maior probabilidade primeiro

    # 3 apostas de 50 dezenas cada (variações de janela)
    aposta1 = idx_sorted_desc[0:50]     # top 50
    aposta2 = idx_sorted_desc[10:60]
    aposta3 = idx_sorted_desc[20:70]

    apostas = [aposta1, aposta2, aposta3]

    # espelhos (complemento em 00-99)
    universo = set(range(100))
    espelhos = []
    for ap in apostas:
        espelhos.append(sorted(list(universo.difference(set(ap)))))

    return apostas, espelhos


def gerar_apostas_errar_tudo(history: List[Set[int]], model: ModelWrapper):
    """
    Usa as MENORES probabilidades da rede para tentar errar tudo.
    """
    probs = gerar_probabilidades_para_proximo(history, model)
    idx_sorted_asc = np.argsort(probs).tolist()

    erro1 = idx_sorted_asc[0:50]
    erro2 = idx_sorted_asc[10:60]
    erro3 = idx_sorted_asc[20:70]

    return [erro1, erro2, erro3]


def treino_incremental_pos_concurso(
    history: List[Set[int]], resultado_set: set[int]
):
    """
    Treino incremental leve, pós-concurso:
    - usa o ÚLTIMO ponto da linha do tempo do histórico
    - calcula sequência + features da dezena
    - faz um passo de treino com o resultado confirmado
    """
    try:
        wrapper = load_model_local()
    except FileNotFoundError:
        logger.warning("Treino incremental ignorado: modelo ainda não treinado (/treinar).")
        return
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo para treino incremental: {e}")
        return

    if len(history) < 2:
        logger.warning("Histórico insuficiente para treino incremental.")
        return

    seq_len = wrapper.seq_len
    net = wrapper.net
    idx = len(history) - 1  # último concurso do histórico

    X_ts_list = []
    X_feat_list = []
    y_list = []

    for dezena in range(100):
        ts = compute_ts_for_dozen(history, idx, dezena, seq_len)
        feats = compute_features_for_dozen(history, idx, dezena)
        X_ts_list.append(ts)
        X_feat_list.append(feats)
        y_list.append(1 if dezena in resultado_set else 0)

    X_ts = np.array(X_ts_list, dtype=np.float32)
    X_feat = np.array(X_feat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # scaler já existente
    X_feat_scaled = (X_feat - wrapper.mean_feat) / wrapper.std_feat

    # passo rápido de ajuste
    net.fit(X_ts, X_feat_scaled, y, epochs=10, batch_size=100)

    # salva e atualiza cache
    save_model(wrapper)
    global _model_cache
    _model_cache = wrapper

    logger.info("Treino incremental pós-concurso concluído (híbrido CNN+MLP).")


def gerar_apostas_oraculo_supremo(
    history: List[Set[int]], model: ModelWrapper
):
    """
    Oráculo Supremo – 6 apostas totalmente independentes:

      1 – Repetição
      2 – Ciclos
      3 – Probabilística real
      4 – Híbrida (prob + ciclos)
      5 – Quentes
      6 – Frias
    """

    if len(history) < 5:
        raise ValueError("Histórico insuficiente para Oráculo Supremo.")

    # ====================================================
    #   PROBABILIDADES DA REDE – NORMALIZADAS
    # ====================================================
    probs = gerar_probabilidades_para_proximo(history, model)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()

    # ====================================================
    #   FREQUÊNCIAS (QUENTES) E ATRASOS (FRIAS)
    # ====================================================
    janela = min(200, len(history))
    freq = np.zeros(100, dtype=np.int32)

    for conc in history[-janela:]:
        for d in conc:
            freq[d] += 1

    atrasos = np.zeros(100, dtype=np.int32)
    for d in range(100):
        gap = 0
        for conc in reversed(history):
            if d in conc:
                break
            gap += 1
        atrasos[d] = gap

    # ====================================================
    #   RUÍDO ADAPTATIVO – muda A CADA EXECUÇÃO
    # ====================================================
    rng = np.random.default_rng()
    ruido = rng.normal(0, 0.05, size=probs.shape)  # ruído um pouco maior p/ variar mais
    probs_ruido = probs + ruido
    probs_ruido = np.clip(probs_ruido, 1e-9, None)
    probs_ruido = probs_ruido / probs_ruido.sum()

    # ====================================================
    #   APOSTA 1 – REPETIÇÃO (prioriza dezenas do último concurso)
    # ====================================================
    ultimo = history[-1]
    cand_rep = sorted(list(ultimo), key=lambda d: probs[d], reverse=True)

    aposta1 = cand_rep[:20]  # 20 repetidas
    restantes = [int(d) for d in np.argsort(-probs) if d not in aposta1]
    aposta1 += restantes[: (50 - len(aposta1))]
    aposta1 = sorted(aposta1)

    # ====================================================
    #   APOSTA 2 – CICLOS (prioriza atrasadas com prob alta)
    # ====================================================
    score_ciclo = (probs ** 0.6) * (1.0 + atrasos / max(atrasos))
    idx_ciclo = np.argsort(score_ciclo)[-50:]
    aposta2 = sorted(idx_ciclo.tolist())

    # ====================================================
    #   APOSTA 3 – PROBABILÍSTICA REAL (amostragem nas probs com ruído)
    # ====================================================
    aposta3 = sorted(rng.choice(100, size=50, replace=False, p=probs_ruido))

    # ====================================================
    #   APOSTA 4 – HÍBRIDA (prob + ciclos não-linear)
    # ====================================================
    score_hibrido = (probs ** 0.5) * (1.0 + atrasos ** 0.4)
    idx_hib = np.argsort(score_hibrido)[-50:]
    aposta4 = sorted(idx_hib.tolist())

    # ====================================================
    #   APOSTA 5 – QUENTES (freq)
    # ====================================================
    aposta5 = sorted(np.argsort(freq)[-50:].tolist())

    # ====================================================
    #   APOSTA 6 – FRIAS (freq baixa + atraso alto)
    # ====================================================
    score_frio = freq * 0.2 + atrasos * 0.8
    aposta6 = sorted(np.argsort(score_frio)[:50].tolist())

    # ====================================================
    #   ESPELHOS
    # ====================================================
    universo = set(range(100))
    apostas = [aposta1, aposta2, aposta3, aposta4, aposta5, aposta6]
    espelhos = [sorted(universo - set(ap)) for ap in apostas]

    return apostas, espelhos


def format_dezenas_sortidas(dezenas):
    return " ".join(f"{d:02d}" for d in sorted(dezenas))


# ----------------------------------------------------
# HANDLERS TELEGRAM
# ----------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🔮 Oráculo Lotomania – Modo C (Híbrido CNN + MLP)\n\n"
        "/treinar - treina ou atualiza a rede neural híbrida\n"
        "/gerar - Oráculo Supremo (6 apostas + 6 espelhos)\n"
        "/errar_tudo - gera 3 apostas tentando errar tudo\n"
        "/confirmar - confronta o resultado oficial com o último bloco gerado\n\n"
        "Mantenha o arquivo lotomania_historico_onehot.csv sempre atualizado."
    )
    await update.message.reply_text(msg)


async def confirmar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /confirmar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94

    1) Lê a última geração salva pelo /gerar ou /errar_tudo
    2) Compara o resultado com as apostas + espelhos
    3) Salva histórico de acertos em CSV
    4) Dispara treino incremental da rede neural híbrida
    """
    texto = (update.message.text or "").strip()
    partes = texto.split()

    if len(partes) < 21:
        await update.message.reply_text(
            "❌ Uso correto:\n"
            "/confirmar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94"
        )
        return

    # ----------------------------------
    # 1) Parse do resultado oficial
    # ----------------------------------
    try:
        dezenas_str = partes[1:]
        dezenas_int = [int(d) for d in dezenas_str]

        # filtra duplicadas e valida faixa
        resultado = []
        for d in dezenas_int:
            if 0 <= d <= 99 and d not in resultado:
                resultado.append(d)

        if len(resultado) != 20:
            await update.message.reply_text(
                f"❌ Informe exatamente 20 dezenas válidas (00–99). Recebi {len(resultado)}."
            )
            return

        resultado_set = set(resultado)

    except ValueError:
        await update.message.reply_text(
            "❌ Não consegui interpretar as dezenas. Use apenas números separados por espaço."
        )
        return

    # ----------------------------------
    # 2) Carrega última geração
    # ----------------------------------
    if not os.path.exists(ULTIMA_GERACAO_PATH):
        await update.message.reply_text(
            "⚠️ Arquivo de última geração não encontrado.\n"
            "Gere um novo bloco com /gerar ou /errar_tudo e depois use /confirmar."
        )
        return

    try:
        with open(ULTIMA_GERACAO_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)

        apostas = dados.get("apostas")
        espelhos = dados.get("espelhos")
        modo = dados.get("modo", "oraculo")  # "oraculo" ou "errar_tudo"

        if not apostas or not espelhos:
            raise ValueError("Dados incompletos na última geração.")

        # garante int nativo
        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

    except Exception:
        logger.exception("Erro ao ler arquivo de última geração.")
        await update.message.reply_text(
            "⚠️ Arquivo de última geração está corrompido ou em formato antigo.\n"
            "Use /gerar novamente para criar um novo bloco de apostas e depois /confirmar."
        )
        return

    # ----------------------------------
    # 3) Calcula acertos por aposta/espelho
    # ----------------------------------
    hits_apostas = []
    hits_espelhos = []

    for ap, esp in zip(apostas_py, espelhos_py):
        hits_apostas.append(len(resultado_set.intersection(ap)))
        hits_espelhos.append(len(resultado_set.intersection(esp)))

    n_apostas = len(hits_apostas)
    n_esp = len(hits_espelhos)

    if n_apostas == 0:
        await update.message.reply_text("⚠️ Não há apostas válidas na última geração.")
        return

    melhor_ap_idx = int(np.argmax(hits_apostas))  # 0..n-1
    melhor_esp_idx = int(np.argmax(hits_espelhos))

    # ----------------------------------
    # 4) Salva histórico de acertos em CSV
    # ----------------------------------
    try:
        existe = os.path.exists(DESEMPENHO_PATH)
        with open(DESEMPENHO_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")

            if not existe:
                header = [
                    "timestamp",
                    "resultado",
                    "acertos_ap1", "acertos_ap2", "acertos_ap3",
                    "acertos_ap4", "acertos_ap5", "acertos_ap6",
                    "acertos_esp1", "acertos_esp2", "acertos_esp3",
                    "acertos_esp4", "acertos_esp5", "acertos_esp6",
                    "melhor_aposta", "melhor_espelho",
                    "modo",
                ]
                writer.writerow(header)

            ts = time.time()
            resultado_txt = " ".join(f"{d:02d}" for d in sorted(resultado))

            # Padding para sempre ter 6 posições
            ha = (hits_apostas + [0] * 6)[:6]
            he = (hits_espelhos + [0] * 6)[:6]

            row = [
                f"{ts:.3f}",
                resultado_txt,
                *[int(h) for h in ha],
                *[int(h) for h in he],
                melhor_ap_idx + 1,
                melhor_esp_idx + 1,
                modo,
            ]
            writer.writerow(row)

        logger.info("Desempenho registrado em %s", DESEMPENHO_PATH)

    except Exception as e_csv:
        logger.exception("Erro ao salvar desempenho em CSV: %s", e_csv)

    # ----------------------------------
    # 5) Treino incremental da rede neural híbrida
    # ----------------------------------
    try:
        history = load_history(HISTORY_PATH)
        treino_incremental_pos_concurso(history, resultado_set)
        txt_treino = "\n🧠 Treino incremental aplicado ao modelo (CNN+MLP)."
    except Exception as e_inc:
        logger.exception("Erro no treino incremental pós-concurso: %s", e_inc)
        txt_treino = "\n⚠️ Não foi possível aplicar o treino incremental (ver logs)."

    # ----------------------------------
    # 6) Resposta para o usuário
    # ----------------------------------
    linhas = []
    linhas.append("✅ Resultado confirmado!")
    linhas.append("Dezenas sorteadas:")
    linhas.append(" ".join(f"{d:02d}" for d in sorted(resultado)))
    linhas.append("")

    if modo == "errar_tudo":
        labels = [f"Aposta erro {i}" for i in range(1, n_apostas + 1)]
    else:
        labels = [
            "Aposta 1 – Repetição",
            "Aposta 2 – Ciclos",
            "Aposta 3 – Probabilística",
            "Aposta 4 – Híbrida",
            "Aposta 5 – Dezenas quentes",
            "Aposta 6 – Dezenas frias",
        ]
        labels = labels[:n_apostas]

    for i in range(n_apostas):
        linhas.append(f"{labels[i]}: {hits_apostas[i]} acertos")
        linhas.append(f"Espelho {i+1}: {hits_espelhos[i]} acertos")
        linhas.append("")

    if modo == "errar_tudo":
        linhas.append(
            f"🏅 Melhor aposta erro: Aposta erro {melhor_ap_idx+1} ({hits_apostas[melhor_ap_idx]} pontos)"
        )
        linhas.append(
            f"🏅 Melhor espelho erro: Espelho erro {melhor_esp_idx+1} ({hits_espelhos[melhor_esp_idx]} pontos)"
        )
    else:
        linhas.append(
            f"🏅 Melhor aposta: {labels[melhor_ap_idx]} ({hits_apostas[melhor_ap_idx]} pontos)"
        )
        linhas.append(
            f"🏅 Melhor espelho: Espelho {melhor_esp_idx+1} ({hits_espelhos[melhor_esp_idx]} pontos)"
        )

    linhas.append(txt_treino)

    await update.message.reply_text("\n".join(linhas).strip())


async def treinar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("🧠 Iniciando treinamento híbrido (CNN + MLP) com o histórico...")

        history = load_history(HISTORY_PATH)

        # Define seq_len padrão (ajuste se quiser mais longo)
        seq_len_default = 40

        # Tenta reaproveitar modelo existente (mesmo seq_len)
        wrapper_antigo = None
        seq_len = min(seq_len_default, len(history) - 1)

        try:
            wrapper_antigo = load_model_local()
            seq_len = wrapper_antigo.seq_len
        except FileNotFoundError:
            wrapper_antigo = None
        except RuntimeError as e_incompat:
            # Modelo antigo (sem CNN) → ignora e treina do zero
            logger.warning(str(e_incompat))
            wrapper_antigo = None
        except Exception as e:
            logger.exception("Erro ao carregar modelo antigo, treinando do zero.")
            wrapper_antigo = None

        X_ts, X_feat, y = build_dataset_hybrid(history, seq_len)
        mean_feat, std_feat = compute_scaler(X_feat)
        X_feat_scaled = (X_feat - mean_feat) / std_feat

        feat_dim = X_feat.shape[1]

        if wrapper_antigo is not None:
            net = wrapper_antigo.net
        else:
            net = HybridCNNMLP(
                seq_len=seq_len,
                feat_dim=feat_dim,
                conv_channels=8,
                kernel_size=5,
                hidden_dim=32,
                lr=0.01,
            )

        net.fit(X_ts, X_feat_scaled, y, epochs=60, batch_size=512)

        wrapper = ModelWrapper(net, mean_feat, std_feat, seq_len)
        save_model(wrapper)
        global _model_cache
        _model_cache = wrapper

        await update.message.reply_text(
            f"✅ Treinamento concluído (híbrido CNN+MLP). Amostras usadas: {len(y)}\n"
            f"seq_len = {seq_len}"
        )

    except Exception as e:
        logger.exception("Erro no treinamento")
        await update.message.reply_text(f"❌ Erro no treinamento: {e}")


async def gerar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        history = load_history(HISTORY_PATH)
        model = load_model_local()

        # Usa o Oráculo Supremo (com modelo híbrido)
        apostas, espelhos = gerar_apostas_oraculo_supremo(history, model)

        # CONVERSÃO: transforma tudo em int nativo do Python
        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

        # Salva última geração para o /confirmar
        try:
            dados = {
                "timestamp": float(time.time()),
                "modo": "oraculo",
                "apostas": apostas_py,
                "espelhos": espelhos_py,
            }

            with open(ULTIMA_GERACAO_PATH, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)

        except Exception as e_save:
            logger.exception(f"Erro ao salvar última geração: {e_save}")

        labels = [
            "Aposta 1 – Repetição",
            "Aposta 2 – Ciclos",
            "Aposta 3 – Probabilística",
            "Aposta 4 – Híbrida",
            "Aposta 5 – Dezenas quentes",
            "Aposta 6 – Dezenas frias",
        ]

        def fmt(lista):
            return " ".join(f"{d:02d}" for d in sorted(lista))

        linhas = ["🔮 Oráculo Supremo – Apostas (Lotomania)\n"]

        for i, (ap, esp) in enumerate(zip(apostas_py, espelhos_py), start=1):
            linhas.append(f"{labels[i-1]}:")
            linhas.append(fmt(ap))
            linhas.append(f"Espelho {i}:")
            linhas.append(fmt(esp))
            linhas.append("")

        texto = "\n".join(linhas).strip()
        await update.message.reply_text(texto)

    except Exception as e:
        logger.exception("Erro ao gerar apostas (Oráculo Supremo)")
        await update.message.reply_text(f"⚠️ Erro ao gerar apostas: {e}")


async def errar_tudo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        history = load_history(HISTORY_PATH)
        model = load_model_local()

        apostas_erro = gerar_apostas_errar_tudo(history, model)

        # Cria espelhos também para errar_tudo
        universo = set(range(100))
        espelhos_erro = [sorted(list(universo - set(ap))) for ap in apostas_erro]

        # CONVERTE para int nativo
        apostas_py = [[int(x) for x in ap] for ap in apostas_erro]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos_erro]

        # Salva como "modo = errar_tudo" para o /confirmar
        try:
            dados = {
                "timestamp": float(time.time()),
                "modo": "errar_tudo",
                "apostas": apostas_py,
                "espelhos": espelhos_py,
            }
            with open(ULTIMA_GERACAO_PATH, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
        except Exception as e_save:
            logger.exception(f"Erro ao salvar última geração (errar_tudo): {e_save}")

        linhas = ["🙃 Apostas para tentar errar tudo\n"]
        for i, ap in enumerate(apostas_py, start=1):
            linhas.append(f"Aposta erro {i}: {format_dezenas_sortidas(ap)}")

        await update.message.reply_text("\n".join(linhas))

    except Exception as e:
        logger.exception("Erro ao gerar apostas de erro")
        await update.message.reply_text(f"❌ Erro ao gerar apostas de erro: {e}")


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Defina a variável de ambiente TELEGRAM_BOT_TOKEN.")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("treinar", treinar_cmd))
    app.add_handler(CommandHandler("gerar", gerar_cmd))
    app.add_handler(CommandHandler("errar_tudo", errar_tudo_cmd))
    app.add_handler(CommandHandler("confirmar", confirmar_cmd))

    logger.info("Bot Lotomania (Oráculo CNN+MLP) iniciado.")
    app.run_polling()


if __name__ == "__main__":
    main()
