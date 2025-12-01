# ORACULO.py – Oráculo Lotomania – Modo C (Híbrido CNN + MLP) – Modo Intensivo + Whitelist + Aprendizado Inteligente

import json
import time
import os
import csv
import logging
import shutil
from typing import List, Set

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ----------------------------------------------------
# Caminhos de arquivos principais
# ----------------------------------------------------
HISTORY_PATH = "lotomania_historico_onehot.csv"   # histórico one-hot (00–99)
MODEL_PATH = "lotomania_model.npz"                # pesos da rede neural (modelo atual)

# Snapshot do melhor modelo já visto (para você poder voltar se quiser)
BEST_MODEL_PATH = "lotomania_model_best.npz"
BEST_SCORE_PATH = "lotomania_best_score.json"

# Arquivo para guardar o último lote gerado (para o /confirmar)
ULTIMA_GERACAO_PATH = "ultima_geracao_oraculo.json"

# Arquivo de telemetria de desempenho
DESEMPENHO_PATH = "desempenho_oraculo.csv"

# Arquivo de whitelist (user_ids autorizados)
WHITELIST_PATH = "whitelist.txt"

# Flag global: se False, /confirmar só valida acertos (não treina o modelo)
TREINO_HABILITADO = True

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Cache simples para whitelist em memória (recarregado a cada uso rápido)
_whitelist_cache: set[int] | None = None

# ----------------------------------------------------
# Funções de controle de acesso (whitelist)
# ----------------------------------------------------


def load_whitelist_ids() -> set[int]:
    """
    Lê o arquivo whitelist.txt e devolve um set de user_ids (int).
    Linhas vazias ou iniciadas por '#' são ignoradas.
    """
    global _whitelist_cache

    # Se já carregado uma vez, reaproveita: o arquivo é pequeno
    # e se você editar o whitelist.txt basta reiniciar o bot.
    if _whitelist_cache is not None:
        return _whitelist_cache

    ids: set[int] = set()
    try:
        with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    ids.add(int(line))
                except ValueError:
                    logger.warning("Linha inválida no whitelist.txt (não é inteiro): %r", line)
    except FileNotFoundError:
        logger.warning(
            "Arquivo de whitelist não encontrado (%s). "
            "Sem ele, ninguém terá acesso a /confirmar ou /treinar.",
            WHITELIST_PATH,
        )

    _whitelist_cache = ids
    return ids


def is_user_whitelisted(update: Update) -> bool:
    user = update.effective_user
    if not user:
        return False
    wl = load_whitelist_ids()
    return user.id in wl


def get_user_label(update: Update) -> str:
    user = update.effective_user
    if not user:
        return "usuário"
    if user.username:
        return f"@{user.username}"
    return f"{user.full_name} (id={user.id})"


# ----------------------------------------------------
# REDE NEURAL HÍBRIDA – CNN 1D + MLP (MODO INTENSIVO)
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
        conv_channels: int = 16,   # antes 8 → agora CNN mais forte
        kernel_size: int = 7,      # antes 5 → kernel maior (mais contexto temporal)
        hidden_dim: int = 64,      # antes 32 → MLP mais robusto
        lr: float = 0.008,         # leve ajuste de learning rate para estabilidade
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

                # Gradiente BCE
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

                # Backprop no pooling:
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
# Snapshot do melhor modelo
# ----------------------------------------------------


def carregar_melhor_info() -> dict:
    """
    Lê o JSON de melhor desempenho: {
        best_hits,
        best_media,
        (opcional) best_pattern,
        (opcional) best_ap_index
    }.
    Se não existir, retorna valores padrão.
    """
    if not os.path.exists(BEST_SCORE_PATH):
        return {
            "best_hits": 0,
            "best_media": 0.0,
            "best_pattern": [],
            "best_ap_index": 0,
        }
    try:
        with open(BEST_SCORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("formato inválido")

        best_hits = int(data.get("best_hits", 0))
        best_media = float(data.get("best_media", 0.0))

        raw_pattern = data.get("best_pattern") or []
        if isinstance(raw_pattern, list):
            best_pattern = [int(d) for d in raw_pattern if 0 <= int(d) <= 99]
        else:
            best_pattern = []

        best_ap_index = int(data.get("best_ap_index", 0))

        return {
            "best_hits": best_hits,
            "best_media": best_media,
            "best_pattern": best_pattern,
            "best_ap_index": best_ap_index,
        }
    except Exception as e:
        logger.warning("Erro ao ler %s: %s. Usando padrão.", BEST_SCORE_PATH, e)
        return {
            "best_hits": 0,
            "best_media": 0.0,
            "best_pattern": [],
            "best_ap_index": 0,
        }


def registrar_melhor_modelo(
    melhor_hits_atual: int,
    media_atual: float,
    aposta_campea: list[int] | None = None,
    idx_campeao: int | None = None,
) -> bool:
    """
    Se o lote atual for melhor que o anterior (por hits e média),
    copia o MODEL_PATH para BEST_MODEL_PATH e atualiza BEST_SCORE_PATH.

    Também registra:
      - best_pattern: dezenas da aposta campeã
      - best_ap_index: índice (1..6) da aposta campeã

    Retorna True se um novo melhor modelo foi salvo.
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning("MODEL_PATH não encontrado; não há modelo para snapshot.")
        return False

    info_ant = carregar_melhor_info()
    best_hits_old = info_ant.get("best_hits", 0)
    best_media_old = info_ant.get("best_media", 0.0)

    improved = False
    if melhor_hits_atual > best_hits_old:
        improved = True
    elif melhor_hits_atual == best_hits_old and media_atual > best_media_old:
        improved = True

    if not improved:
        return False

    try:
        shutil.copy2(MODEL_PATH, BEST_MODEL_PATH)

        info_new = {
            "best_hits": int(melhor_hits_atual),
            "best_media": float(media_atual),
        }

        if aposta_campea is not None:
            # salva padrão da aposta campeã (ordenado, sem duplicatas)
            pattern = sorted({int(d) for d in aposta_campea if 0 <= int(d) <= 99})
            info_new["best_pattern"] = pattern

        if idx_campeao is not None:
            info_new["best_ap_index"] = int(idx_campeao)

        with open(BEST_SCORE_PATH, "w", encoding="utf-8") as f:
            json.dump(info_new, f, ensure_ascii=False, indent=2)

        logger.info(
            "Novo melhor modelo registrado: %d acertos, média %.2f. Snapshot salvo em %s",
            melhor_hits_atual,
            media_atual,
            BEST_MODEL_PATH,
        )
        return True
    except Exception as e:
        logger.exception("Falha ao salvar snapshot do melhor modelo: %s", e)
        return False


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
    Modo Erro Supremo – usa a mesma base híbrida (probs + freq + atraso),
    mas invertida para tentar maximizar o ERRO.

    Gera 3 apostas de 50 dezenas cada, usando janelas deslizantes sobre o
    ranking de score_erro, garantindo que NENHUMA aposta venha vazia.
    """
    if len(history) < 5:
        raise ValueError("Histórico insuficiente para gerar apostas de erro.")

    # Probabilidades da rede (base)
    probs = gerar_probabilidades_para_proximo(history, model)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()

    n_hist = len(history)

    # Frequências em múltiplas janelas (mesma ideia do Oráculo Supremo)
    janela_long = min(200, n_hist)
    freq_long = np.zeros(100, dtype=np.int32)
    for conc in history[-janela_long:]:
        for d in conc:
            freq_long[d] += 1

    janela10 = min(10, n_hist)
    freq10 = np.zeros(100, dtype=np.int32)
    for conc in history[-janela10:]:
        for d in conc:
            freq10[d] += 1

    # Atrasos
    atrasos = np.zeros(100, dtype=np.int32)
    for d in range(100):
        gap = 0
        for conc in reversed(history):
            if d in conc:
                break
            gap += 1
        atrasos[d] = gap

    def _norm(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        maxv = float(arr.max())
        if maxv <= 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        return arr / maxv

    probs_n = _norm(probs)
    freq_long_n = _norm(freq_long)
    freq10_n = _norm(freq10)
    atraso_n = _norm(atrasos)

    # Score de ERRO (híbrido invertido):
    #   - (1 - probs_n): mais peso para dezenas que a rede acha ruins
    #   - (1 - freq_long_n) e (1 - freq10_n): pouco saíram
    #   - atraso_n (opcional) pode ser somado se quiser forçar ainda mais atraso
    score_erro = (
        0.6 * (1.0 - probs_n) +
        0.2 * (1.0 - freq_long_n) +
        0.2 * (1.0 - freq10_n)
        # + 0.1 * atraso_n  # se quiser enfatizar ainda mais as muito atrasadas
    )

    # Ordena do "melhor para errar" para o "pior para errar"
    idx_erro_desc = np.argsort(-score_erro)

    # 3 apostas com janelas deslizantes (sempre 50 dezenas cada)
    aposta1 = [int(d) for d in idx_erro_desc[0:50]]
    aposta2 = [int(d) for d in idx_erro_desc[10:60]]
    aposta3 = [int(d) for d in idx_erro_desc[20:70]]

    apostas_erro: List[List[int]] = [
        sorted(aposta1),
        sorted(aposta2),
        sorted(aposta3),
    ]

    return apostas_erro


def treino_incremental_pos_concurso(
    history: List[Set[int]],
    resultado_set: set[int],
    epochs: int = 35,
    batch_size: int = 64,
):
    """
    Treino incremental intensivo, pós-concurso:
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

    net.fit(X_ts, X_feat_scaled, y, epochs=epochs, batch_size=batch_size)

    # salva e atualiza cache
    save_model(wrapper)
    global _model_cache
    _model_cache = wrapper

    logger.info(
        "Treino incremental pós-concurso concluído (modo intensivo CNN+MLP). "
        "epochs=%d batch_size=%d",
        epochs,
        batch_size,
    )


def gerar_apostas_oraculo_supremo(
    history: List[Set[int]], model: ModelWrapper
):
    """
    Oráculo Supremo – 6 apostas totalmente independentes e diversificadas:

      1 – Repetição inteligente (poucas repetidas + top prob)
      2 – Ciclos (atraso forte + probabilidade)
      3 – Probabilística real (amostragem nas probs com ruído)
      4 – Híbrida (CNN/MLP + freq + ciclos, evitando reuso)
      5 – Quentes (multi-janela 10/30/200)
      6 – Frias (baixa freq + atraso alto, evitando reuso)

    Após existir um recorde >= 15 acertos, a aposta campeã passa
    a ser gerada com âncora 70/30:
      - ~70% das dezenas da campeã fixas
      - ~30% de variação leve guiada pelas probabilidades da rede
    """

    if len(history) < 5:
        raise ValueError("Histórico insuficiente para Oráculo Supremo.")

    # ====================================================
    #   PROBABILIDADES DA REDE – NORMALIZADAS (base)
    # ====================================================
    probs = gerar_probabilidades_para_proximo(history, model)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()

    n_hist = len(history)

    # ====================================================
    #   FREQUÊNCIAS (QUENTES) EM MÚLTIPLAS JANELAS
    # ====================================================
    janela_long = min(200, n_hist)
    freq_long = np.zeros(100, dtype=np.int32)
    for conc in history[-janela_long:]:
        for d in conc:
            freq_long[d] += 1

    janela10 = min(10, n_hist)
    freq10 = np.zeros(100, dtype=np.int32)
    for conc in history[-janela10:]:
        for d in conc:
            freq10[d] += 1

    janela30 = min(30, n_hist)
    freq30 = np.zeros(100, dtype=np.int32)
    for conc in history[-janela30:]:
        for d in conc:
            freq30[d] += 1

    # ====================================================
    #   ATRASOS (FRIAS)
    # ====================================================
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
    ruido = rng.normal(0, 0.08, size=probs.shape)  # ruído um pouco maior p/ variar mais
    probs_ruido = probs + ruido
    probs_ruido = np.clip(probs_ruido, 1e-9, None)
    probs_ruido = probs_ruido / probs_ruido.sum()

    # ----------------------------------------------------
    # Normalizações auxiliares
    # ----------------------------------------------------
    def _norm(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        maxv = float(arr.max())
        if maxv <= 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        return arr / maxv

    freq_long_n = _norm(freq_long)
    freq10_n = _norm(freq10)
    freq30_n = _norm(freq30)
    atraso_n = _norm(atrasos)
    probs_n = _norm(probs)

    # ====================================================
    #   APOSTA 1 – REPETIÇÃO INTELIGENTE
    #   (poucas repetidas + top CNN/MLP)
    # ====================================================
    ultimo = history[-1]
    cand_rep_ordenados = sorted(list(ultimo), key=lambda d: probs[d], reverse=True)
    n_rep = min(12, len(cand_rep_ordenados))  # limita repetição para não "colar" demais
    base_ap1 = cand_rep_ordenados[:n_rep]

    restantes_ordem = [int(d) for d in np.argsort(-probs) if d not in base_ap1]
    aposta1 = base_ap1 + restantes_ordem[: (50 - len(base_ap1))]
    aposta1 = sorted(aposta1)

    # ====================================================
    #   APOSTA 2 – CICLOS (atraso forte + probabilidade)
    #   Diversificada em relação à Aposta 1
    # ====================================================
    ciclo_score = (probs_n ** 0.7) * (1.0 + atraso_n * 1.5)
    idx_ciclo_pool = np.argsort(ciclo_score)[-70:]  # pool maior

    aposta2 = []
    usados_ap1 = set(aposta1)
    for d in reversed(idx_ciclo_pool):
        if d in usados_ap1:
            continue
        aposta2.append(int(d))
        if len(aposta2) == 50:
            break

    if len(aposta2) < 50:
        usados = set(aposta2)
        for d in np.argsort(ciclo_score)[::-1]:
            if d in usados:
                continue
            aposta2.append(int(d))
            if len(aposta2) == 50:
                break

    aposta2 = sorted(aposta2)

    # ====================================================
    #   APOSTA 3 – PROBABILÍSTICA REAL
    #   (amostragem nas probs com ruído + tempering)
    # ====================================================
    temp = 0.85
    probs_temp = probs_ruido ** temp
    probs_temp = np.clip(probs_temp, 1e-9, None)
    probs_temp = probs_temp / probs_temp.sum()

    aposta3 = sorted(rng.choice(100, size=50, replace=False, p=probs_temp))

    # ====================================================
    #   APOSTA 4 – HÍBRIDA (CNN/MLP + freq + atraso)
    #   Evita reuso pesado de dezenas já presentes nas 1–3
    # ====================================================
    score_hibrido = (
        0.6 * probs_n +
        0.2 * atraso_n +
        0.2 * freq_long_n
    )
    idx_hib_pool = np.argsort(score_hibrido)[-80:]

    usados_1a3 = set(aposta1) | set(aposta2) | set(aposta3)
    aposta4 = []
    for d in reversed(idx_hib_pool):
        if d in usados_1a3:
            continue
        aposta4.append(int(d))
        if len(aposta4) == 50:
            break

    if len(aposta4) < 50:
        usados = set(aposta4)
        for d in np.argsort(score_hibrido)[::-1]:
            if d in usados:
                continue
            aposta4.append(int(d))
            if len(aposta4) == 50:
                break

    aposta4 = sorted(aposta4)

    # ====================================================
    #   APOSTA 5 – QUENTES (multi-janela 10/30/200)
    #   f10 > f30 > f200
    # ====================================================
    quente_score = 0.5 * freq10_n + 0.3 * freq30_n + 0.2 * freq_long_n
    idx_quentes_pool = np.argsort(quente_score)[-80:]

    usados_1a4 = usados_1a3 | set(aposta4)
    aposta5 = []
    for d in reversed(idx_quentes_pool):
        if d in usados_1a4:
            continue
        aposta5.append(int(d))
        if len(aposta5) == 50:
            break

    if len(aposta5) < 50:
        usados = set(aposta5)
        for d in np.argsort(quente_score)[::-1]:
            if d in usados:
                continue
            aposta5.append(int(d))
            if len(aposta5) == 50:
                break

    aposta5 = sorted(aposta5)

    # ====================================================
    #   APOSTA 6 – FRIAS (baixa freq + atraso alto)
    #   Evita reuso das demais, com ruído leve para não "travar"
    # ====================================================
    frias_score_base = 0.6 * atraso_n + 0.4 * (1.0 - freq_long_n)

    # Ruído bem leve só nas frias, para mudar um pouco a cada /gerar,
    # sem descaracterizar o conceito de "fria".
    ruido_frias = rng.normal(0, 0.02, size=frias_score_base.shape)
    frias_score = frias_score_base + ruido_frias

    idx_frias_pool = np.argsort(frias_score)[-80:]  # maiores scores = mais "frias relevantes"

    usados_1a5 = usados_1a4 | set(aposta5)
    aposta6 = []
    for d in reversed(idx_frias_pool):
        if d in usados_1a5:
            continue
        aposta6.append(int(d))
        if len(aposta6) == 50:
            break

    if len(aposta6) < 50:
        usados = set(aposta6)
        for d in np.argsort(frias_score)[::-1]:
            if d in usados:
                continue
            aposta6.append(int(d))
            if len(aposta6) == 50:
                break

    aposta6 = sorted(aposta6)

    # ====================================================
    #   ANCORAGEM 70/30 NA APOSTA CAMPEÃ (se houver recorde >= 15)
    # ====================================================
    apostas = [aposta1, aposta2, aposta3, aposta4, aposta5, aposta6]

    best_info = carregar_melhor_info()
    best_hits = int(best_info.get("best_hits", 0))
    best_pattern = best_info.get("best_pattern") or []
    best_ap_index = int(best_info.get("best_ap_index", 0))

    if best_hits >= 15 and isinstance(best_pattern, list) and len(best_pattern) >= 10:
        idx_anchor = best_ap_index - 1  # converter 1..6 -> 0..5
        if 0 <= idx_anchor < len(apostas):
            # garante conjunto válido de dezenas
            campea = sorted({int(d) for d in best_pattern if 0 <= int(d) <= 99})
            if len(campea) > 0:
                n_total = 50
                frac_fixo = 0.70
                n_fixo = min(len(campea), int(round(n_total * frac_fixo)))

                # fixa as dezenas da campeã com maior probabilidade atual
                campea_ordenada = sorted(campea, key=lambda d: probs[d], reverse=True)
                fixas = campea_ordenada[:n_fixo]
                fixas_set = set(fixas)

                # preenche o restante com dezenas de maior probabilidade fora do conjunto fixo
                variavel = []
                for d in np.argsort(-probs):
                    d_int = int(d)
                    if d_int in fixas_set:
                        continue
                    variavel.append(d_int)
                    if len(fixas) + len(variavel) == n_total:
                        break

                aposta_ancorada = sorted(fixas + variavel)
                apostas[idx_anchor] = aposta_ancorada

    # ====================================================
    #   ESPELHOS
    # ====================================================
    universo = set(range(100))
    espelhos = [sorted(universo - set(ap)) for ap in apostas]

    return apostas, espelhos



def format_dezenas_sortidas(dezenas):
    return " ".join(f"{d:02d}" for d in sorted(dezenas))


# ----------------------------------------------------
# HANDLERS TELEGRAM
# ----------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🔮 Oráculo Lotomania – Modo C (Híbrido CNN + MLP – Intensivo)\n\n"
        "/treinar - treina ou atualiza a rede neural híbrida (treino forte) "
        "(RESTRITO à whitelist)\n"
        "/gerar - Oráculo Supremo (6 apostas + 6 espelhos)\n"
        "/errar_tudo - gera 3 apostas tentando errar tudo\n"
        "/confirmar - confronta o resultado oficial com o último bloco gerado, "
        "registra desempenho e, se habilitado, aplica treino incremental (RESTRITO à whitelist)\n\n"
        "Mantenha o arquivo lotomania_historico_onehot.csv sempre atualizado.\n"
        f"Modo treino habilitado: {'SIM' if TREINO_HABILITADO else 'NÃO (apenas avaliação)'}"
    )
    await update.message.reply_text(msg)


async def confirmar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /confirmar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94

    1) Lê a última geração salva pelo /gerar ou /errar_tudo
    2) Garante que o mesmo usuário que gerou é quem está confirmando
    3) Compara o resultado com as apostas + espelhos
    4) Salva histórico de acertos em CSV
    5) (Opcional) Dispara treino incremental INTENSIVO da rede neural híbrida
    6) (Opcional) Atualiza snapshot do melhor modelo
       + Congelamento rígido 15+ (Opção A):
         - Enquanto nenhum recorde tiver 15+, todos os lotes treinam normal
         - Após um recorde com 15+ acertos, o modelo fica TRAVADO
         - Só volta a treinar se surgir um novo lote que supere o recorde
    """
    # ------------------------------------------------
    # 0) Verifica se usuário está na whitelist
    # ------------------------------------------------
    if not is_user_whitelisted(update):
        usuario = get_user_label(update)
        await update.message.reply_text(
            f"⚠️ {usuario}, você não tem permissão para usar /confirmar.\n"
            "Apenas o proprietário (user_id presente em whitelist.txt) pode confirmar resultados "
            "e ajustar o modelo."
        )
        return

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
        user_id_gerador = dados.get("user_id")  # quem gerou o bloco

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
    # 2.1) Garante que o bloco foi gerado pelo mesmo usuário que está confirmando
    # ----------------------------------
    current_user = update.effective_user
    current_id = current_user.id if current_user else None

    if user_id_gerador is not None and current_id is not None and current_id != user_id_gerador:
        await update.message.reply_text(
            "⚠️ O último bloco de apostas foi gerado por outro usuário.\n"
            "Gere um novo bloco com /gerar antes de usar /confirmar, para treinar apenas em cima "
            "das apostas que você mesmo gerou."
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

    melhor_hits = int(hits_apostas[melhor_ap_idx])
    media_hits = float(sum(hits_apostas) / n_apostas)

    # ----------------------------------
    # 3.1) Lê recorde histórico para aplicar congelamento rígido 15+
    # ----------------------------------
    best_info = carregar_melhor_info()
    best_hits_ref = int(best_info.get("best_hits", 0))
    best_media_ref = float(best_info.get("best_media", 0.0))

    # Congelamento rígido (Opção A):
    # - Só entra em modo travado depois que existir um recorde >= 15
    # - Enquanto recorde >= 15 E o lote atual NÃO superar esse recorde → NÃO treina
    locked_rigid = (best_hits_ref >= 15) and (melhor_hits <= best_hits_ref)

    # ----------------------------------
    # 3.2) Classificação do lote (ajuste de epochs)
    #      (texto atualizado para alinhar com o congelamento 15+)
    # ----------------------------------
    if melhor_hits >= 15:
        classe_lote = (
            "Lote EXCELENTE — atingiu 15+ acertos (zona de congelamento rígido)."
        )
        epochs_inc = 20
    elif melhor_hits >= 11:
        classe_lote = "Lote forte — reforço mais intenso aplicado nas dezenas-chave."
        epochs_inc = 28
    elif melhor_hits >= 9:
        classe_lote = "Lote mediano — ajuste normal aplicado."
        epochs_inc = 22
    else:
        classe_lote = "Lote fraco — ajuste suave (treino mais cauteloso)."
        epochs_inc = 16

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
                    "melhor_hits", "media_hits",
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
                melhor_hits,
                f"{media_hits:.2f}",
            ]
            writer.writerow(row)

        logger.info("Desempenho registrado em %s", DESEMPENHO_PATH)

    except Exception as e_csv:
        logger.exception("Erro ao salvar desempenho em CSV: %s", e_csv)

    # ----------------------------------
    # 5) Treino incremental (INTENSIVO) com congelamento rígido 15+
    # ----------------------------------
    if TREINO_HABILITADO and not locked_rigid:
        # 🔓 Modo livre / destravado:
        # - Ainda não existe recorde 15+
        #   OU
        # - Esse lote SUPEROU o recorde atual (novo recorde)
        try:
            history = load_history(HISTORY_PATH)
        except Exception as e_hist:
            logger.exception("Erro ao carregar histórico para treino incremental: %s", e_hist)
            history = None

        if history is not None:
            try:
                treino_incremental_pos_concurso(
                    history,
                    resultado_set,
                    epochs=epochs_inc,
                    batch_size=64,
                )
                txt_treino = (
                    f"\n🧠 Treino incremental INTENSIVO aplicado ao modelo (CNN+MLP).\n"
                    f"   • Melhor aposta do lote: {melhor_hits} acertos\n"
                    f"   • Média do lote: {media_hits:.2f} acertos\n"
                    f"   • Intensidade de treino usada: {epochs_inc} epochs (modo online)"
                )
            except Exception as e_inc:
                logger.exception("Erro no treino incremental pós-concurso: %s", e_inc)
                txt_treino = "\n⚠️ Não foi possível aplicar o treino incremental (ver logs)."
        else:
            txt_treino = "\n⚠️ Não foi possível carregar o histórico para treino incremental."
    elif TREINO_HABILITADO and locked_rigid:
        # 🧊 Modo TRAVADO (recorde 15+ já existe e este lote NÃO o superou)
        txt_treino = (
            "\n🧊 Congelamento rígido 15+ ATIVO.\n"
            f"   • Recorde atual: {best_hits_ref} acertos "
            "(modelo referência salvo em 'lotomania_model_best.npz').\n"
            "   • Este /confirmar NÃO alterou o modelo — apenas registrou desempenho em CSV.\n"
            "   • O treino só será liberado novamente quando UM NOVO LOTE superar esse recorde."
        )
    else:
        # TREINO_HABILITADO = False → modo avaliação pura
        txt_treino = (
            "\nℹ️ Modo avaliação: /confirmar NÃO está ajustando o modelo "
            "(apenas registrando o desempenho em CSV)."
        )

    # ----------------------------------
    # 5.1) Atualiza snapshot do melhor modelo (se houver melhoria)
    #      (usa o estado ATUAL do modelo; quando travado, nada muda aqui)
    # ----------------------------------
    txt_best = ""
    if TREINO_HABILITADO:
        try:
            if registrar_melhor_modelo(
                melhor_hits,
                media_hits,
                aposta_campea=apostas_py[melhor_ap_idx],
                idx_campeao=(melhor_ap_idx + 1),
            ):
                txt_best = (
                    "\n🏆 Este lote superou o melhor desempenho anterior.\n"
                    "   Modelo atual salvo como 'lotomania_model_best.npz' "
                    "e métricas atualizadas em 'lotomania_best_score.json'."
                )
        except Exception as e_best:
            logger.exception("Erro ao registrar snapshot do melhor modelo: %s", e_best)
            txt_best = "\n⚠️ Não foi possível salvar snapshot do melhor modelo (ver logs)."

    # ----------------------------------
    # 6) Resposta para o usuário
    # ----------------------------------
    linhas = []
    linhas.append("✅ Resultado confirmado!")
    linhas.append("Dezenas sorteadas:")
    linhas.append(" ".join(f"{d:02d}" for d in sorted(resultado)))
    linhas.append("")
    linhas.append(f"Melhor aposta do lote: {melhor_hits} acertos")
    linhas.append(f"Média de acertos do lote: {media_hits:.2f}")
    linhas.append(classe_lote)
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
    if txt_best:
        linhas.append(txt_best)

    await update.message.reply_text("\n".join(linhas).strip())


async def treinar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Restrito à whitelist
    if not is_user_whitelisted(update):
        usuario = get_user_label(update)
        await update.message.reply_text(
            f"⚠️ {usuario}, você não tem permissão para usar /treinar.\n"
            "Apenas o proprietário (user_id presente em whitelist.txt) pode treinar o modelo."
        )
        return

    try:
        await update.message.reply_text(
            "🧠 Iniciando treinamento híbrido (CNN + MLP – modo intensivo) com o histórico..."
        )

        history = load_history(HISTORY_PATH)

        # seq_len mais longo para capturar mais contexto temporal
        seq_len_default = 60

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
            # CNN mais forte + MLP maior em modo intensivo
            net = HybridCNNMLP(
                seq_len=seq_len,
                feat_dim=feat_dim,
                conv_channels=16,
                kernel_size=7,
                hidden_dim=64,
                lr=0.008,
            )

        # TREINO INTENSIVO GLOBAL
        net.fit(X_ts, X_feat_scaled, y, epochs=100, batch_size=512)

        wrapper = ModelWrapper(net, mean_feat, std_feat, seq_len)
        save_model(wrapper)
        global _model_cache
        _model_cache = wrapper

        await update.message.reply_text(
            f"✅ Treinamento concluído (híbrido CNN+MLP – modo intensivo).\n"
            f"Amostras usadas: {len(y)}\n"
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
            user = update.effective_user
            dados = {
                "timestamp": float(time.time()),
                "modo": "oraculo",
                "apostas": apostas_py,
                "espelhos": espelhos_py,
                "user_id": user.id if user else None,  # quem gerou este bloco
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
            user = update.effective_user
            dados = {
                "timestamp": float(time.time()),
                "modo": "errar_tudo",
                "apostas": apostas_py,
                "espelhos": espelhos_py,
                "user_id": user.id if user else None,  # quem gerou este bloco
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

    logger.info("Bot Lotomania (Oráculo CNN+MLP – modo intensivo + whitelist + 70/30 campeão) iniciado.")
    app.run_polling()


if __name__ == "__main__":
    main()
