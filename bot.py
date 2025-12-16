# lotomania_bot.py – Oráculo Lotomania – Modo C (Híbrido CNN + MLP) – Modo Intensivo + Whitelist + Aprendizado Inteligente

import json
import time
import os
import csv
import logging
import shutil
from typing import List, Set, Tuple, Dict
from datetime import datetime
from collections import defaultdict, Counter

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

# Arquivo para registro de penalidades (aprendizado negativo)
PENALIDADES_PATH = "penalidades_oraculo.json"

# ----------------------------------------------------
# CONFIGURAÇÃO GLOBAL DE APRENDIZADO
# ----------------------------------------------------
INTENSIVE_LEARNING = True  # deixar True para modo agressivo de treino
TREINO_HABILITADO = True  # Flag global: se False, /confirmar só valida acertos (não treina o modelo)

# Configurações de penalidade
PENALIDADE_ERRO = 0.3  # Penalidade por escolher dezena que saiu (0.0 a 1.0)
RECOMPENSA_ACERTO_ERRAR = 0.2  # Recompensa por escolher dezena que NÃO saiu
MAX_PENALIDADE = 3.0  # Penalidade máxima acumulada
DECAIMENTO_PENALIDADE = 0.95  # Decaimento das penalidades a cada concurso (95%)

# hiperparâmetros base (modo normal)
BASE_CONV_CHANNELS = 8
BASE_KERNEL_SIZE = 5
BASE_HIDDEN_DIM = 32
BASE_LR = 0.01

# hiperparâmetros reforçados (modo intensivo)
INT_CONV_CHANNELS = 16
INT_KERNEL_SIZE = 7
INT_HIDDEN_DIM = 64
INT_LR = 0.005

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Cache simples para whitelist em memória (recarregado a cada uso rápido)
_whitelist_cache: set[int] | None = None

# Cache de penalidades em memória
_penalidades_cache: Dict[int, float] = None

# ----------------------------------------------------
# FUNÇÕES DE PENALIDADES (APRENDIZADO NEGATIVO)
# ----------------------------------------------------

def carregar_penalidades() -> Dict[int, float]:
    """
    Carrega as penalidades acumuladas por dezena.
    Penalidade positiva = dezena que TEM saído (deve ser evitada)
    Penalidade negativa = dezena que NÃO tem saído (deve ser preferida)
    """
    global _penalidades_cache
    
    if _penalidades_cache is not None:
        return _penalidades_cache
    
    penalidades = defaultdict(float)
    
    if os.path.exists(PENALIDADES_PATH):
        try:
            with open(PENALIDADES_PATH, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                if isinstance(dados, dict):
                    for dezena_str, valor in dados.items():
                        try:
                            dezena = int(dezena_str)
                            if 0 <= dezena <= 99:
                                penalidades[dezena] = float(valor)
                        except ValueError:
                            continue
        except Exception as e:
            logger.warning(f"Erro ao carregar penalidades: {e}")
    
    _penalidades_cache = penalidades
    return penalidades

def salvar_penalidades(penalidades: Dict[int, float]):
    """
    Salva as penalidades no arquivo JSON.
    """
    try:
        with open(PENALIDADES_PATH, 'w', encoding='utf-8') as f:
            # Converte chaves para string para serialização JSON
            dados = {str(k): v for k, v in penalidades.items()}
            json.dump(dados, f, ensure_ascii=False, indent=2)
        
        global _penalidades_cache
        _penalidades_cache = penalidades
        logger.info(f"Penalidades salvas em {PENALIDADES_PATH}")
    except Exception as e:
        logger.error(f"Erro ao salvar penalidades: {e}")

def aplicar_penalidades_apos_resultado(resultado_set: Set[int], modo: str = "errar_tudo"):
    """
    Aplica penalidades após um resultado:
    - Penaliza dezenas que SAÍRAM no resultado (se estamos tentando errar)
    - Recompensa dezenas que NÃO saíram (se estamos tentando errar)
    
    O contrário para modo normal (oraculo).
    """
    penalidades = carregar_penalidades()
    
    todas_dezenas = set(range(100))
    dezenas_nao_sairam = todas_dezenas - resultado_set
    
    if modo == "errar_tudo" or modo == "refino":
        # MODE ERRAR_TUDO / REFINO:
        # Penaliza dezenas que SAÍRAM (erramos ao escolhê-las)
        # Recompensa dezenas que NÃO saíram (acertamos em não escolhê-las)
        for dezena in resultado_set:
            penalidades[dezena] += PENALIDADE_ERRO
            # Limita a penalidade máxima
            penalidades[dezena] = min(penalidades[dezena], MAX_PENALIDADE)
        
        for dezena in dezenas_nao_sairam:
            penalidades[dezena] -= RECOMPENSA_ACERTO_ERRAR
            penalidades[dezena] = max(penalidades[dezena], -MAX_PENALIDADE)
    
    else:
        # MODO ORACULO (tentar acertar):
        # Penaliza dezenas que NÃO saíram (erramos ao escolhê-las)
        # Recompensa dezenas que SAÍRAM (acertamos em escolhê-las)
        for dezena in dezenas_nao_sairam:
            penalidades[dezena] += PENALIDADE_ERRO * 0.5  # Penalidade menor para modo normal
        
        for dezena in resultado_set:
            penalidades[dezena] -= RECOMPENSA_ACERTO_ERRAR * 0.5
    
    # Aplica decaimento para todas as penalidades (esquece lentamente)
    for dezena in list(penalidades.keys()):
        penalidades[dezena] *= DECAIMENTO_PENALIDADE
        # Remove penalidades muito pequenas
        if abs(penalidades[dezena]) < 0.01:
            del penalidades[dezena]
    
    salvar_penalidades(penalidades)
    logger.info(f"Penalidades aplicadas após resultado (modo: {modo})")

def aplicar_decaimento_penalidades():
    """
    Aplica decaimento regular nas penalidades (chamar periodicamente).
    """
    penalidades = carregar_penalidades()
    if penalidades:
        for dezena in list(penalidades.keys()):
            penalidades[dezena] *= DECAIMENTO_PENALIDADE
            if abs(penalidades[dezena]) < 0.01:
                del penalidades[dezena]
        salvar_penalidades(penalidades)

def obter_score_com_penalidades(scores_base: np.ndarray) -> np.ndarray:
    """
    Ajusta os scores base com as penalidades.
    Para modo errar_tudo: scores_ajustados = scores_base + penalidades
    (penalidades positivas aumentam o score = mais chance de ser escolhida para errar)
    """
    penalidades = carregar_penalidades()
    
    scores_ajustados = scores_base.copy()
    
    for dezena, penalidade in penalidades.items():
        if 0 <= dezena <= 99:
            scores_ajustados[dezena] += penalidade
    
    return scores_ajustados

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
        conv_channels: int | None = None,
        kernel_size: int | None = None,
        hidden_dim: int | None = None,
        lr: float | None = None,
        seed: int = None,
    ):
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        if INTENSIVE_LEARNING:
            self.conv_channels = conv_channels or INT_CONV_CHANNELS
            self.kernel_size = kernel_size or INT_KERNEL_SIZE
            self.hidden_dim = hidden_dim or INT_HIDDEN_DIM
            self.lr = lr or INT_LR
        else:
            self.conv_channels = conv_channels or BASE_CONV_CHANNELS
            self.kernel_size = kernel_size or BASE_KERNEL_SIZE
            self.hidden_dim = hidden_dim or BASE_HIDDEN_DIM
            self.lr = lr or BASE_LR

        # Usa seed baseada no tempo atual para variar a cada execução
        if seed is None:
            seed = int(time.time() * 1000) % 1000000
        rng = np.random.default_rng(seed)

        # Convolução 1D: kernel_size x conv_channels
        self.Wc = rng.normal(0, 0.1, size=(self.kernel_size, self.conv_channels)).astype(
            np.float32
        )
        self.bc = np.zeros((self.conv_channels,), dtype=np.float32)

        # MLP (entrada = features CNN + features manuais)
        in_dim = self.conv_channels + feat_dim
        self.W1 = rng.normal(0, 0.1, size=(in_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, size=(self.hidden_dim, 1)).astype(np.float32)
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

        for epoch in range(epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            X_ts_sh = X_ts[idx]
            X_feat_sh = X_feat[idx]
            y_sh = y[idx]

            total_loss = 0.0
            n_batches = 0

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

                # Calcula loss para logging
                loss = np.mean((y_hat - yb) ** 2)
                total_loss += loss
                n_batches += 1

            if n_batches > 0:
                avg_loss = total_loss / n_batches
                logger.debug(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

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
      - y      : 1 se a dezena saiu no próximo concurso, 0 se não
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
    """
    Salva o modelo com todos os pesos e metadados.
    """
    try:
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
        logger.info(f"Modelo salvo em {path}")
    except Exception as e:
        logger.exception(f"Erro ao salvar modelo em {path}: {e}")
        raise


def load_model_local(path: str = MODEL_PATH) -> ModelWrapper:
    """
    Carrega o modelo do disco, com cache em memória.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(path):
        raise FileNotFoundError("Modelo ainda não treinado. Use /treinar.")

    try:
        data = np.load(path)

        # Verifica se é modelo antigo (sem Wc) e força recriação
        if "Wc" not in data.files:
            raise RuntimeError(
                "Modelo salvo é de versão antida (sem CNN). "
                "Apague o arquivo 'lotomania_model.npz' e roda /treinar novamente."
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
        logger.info(f"Modelo carregado de {path}")
        return wrapper
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo de {path}: {e}")
        raise


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


def treino_incremental_pos_concurso(
    history: List[Set[int]], resultado_set: set[int], modo: str = "oraculo"
):
    """
    Treino incremental pós-concurso COM APRENDIZADO DE PENALIDADES (VERSÃO CORRIGIDA).

    Ideia correta:
    - O modelo foi treinado com a lógica: estado do concurso i  ->  resultado do concurso i+1
    - No incremental, fazemos a MESMA COISA, mas só para a transição mais recente.

    Ou seja:
    - Usamos o concurso ANTERIOR como base (idx_base = len(history) - 2)
    - Usamos o resultado NOVO (resultado_set) como target.
    - Mantemos os pesos atuais e continuamos o treinamento (não reseta).
    """
    try:
        wrapper = load_model_local()
    except FileNotFoundError:
        # Só dá esse erro se você NUNCA rodou /treinar pelo menos uma vez.
        logger.warning(
            "Treino incremental ignorado: modelo ainda não treinado (/treinar inicial necessário apenas uma vez)."
        )
        return
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo para treino incremental: {e}")
        return

    # Precisamos de pelo menos 2 concursos no histórico:
    # penúltimo = estado, último = resultado que você está confirmando.
    if len(history) < 2:
        logger.warning("Histórico insuficiente para treino incremental (menos de 2 concursos).")
        return

    seq_len = wrapper.seq_len
    net = wrapper.net

    # ------------------------------------------------
    # DEFINIÇÃO DO PONTO DE TREINO
    # ------------------------------------------------
    # Supondo que o CSV já contém o NOVO concurso:
    #   idx_final = índice do último concurso do histórico
    #   idx_base  = concurso imediatamente anterior (estado)
    idx_final = len(history) - 1
    idx_base = idx_final - 1

    if idx_base < 0:
        logger.warning("Não há concurso anterior suficiente para treino incremental.")
        return

    # Se o histórico ainda for muito curto pra janela seq_len, evita quebrar o modelo.
    if len(history) < seq_len + 1:
        logger.warning(
            "Histórico ainda menor que seq_len+1 (len(history)=%d, seq_len=%d). "
            "Treino incremental pulado para evitar inconsistência.",
            len(history),
            seq_len,
        )
        return

    X_ts_list: list[list[float]] = []
    X_feat_list: list[list[float]] = []
    y_list: list[int] = []

    # ------------------------------------------------
    # CONSTRÓI AMOSTRAS PARA TODAS AS DEZENAS (00–99)
    # NO PONTO idx_base, COM TARGET = resultado_set
    # ------------------------------------------------
    for dezena in range(100):
        ts = compute_ts_for_dozen(history, idx_base, dezena, seq_len)
        feats = compute_features_for_dozen(history, idx_base, dezena)
        X_ts_list.append(ts)
        X_feat_list.append(feats)
        y_list.append(1 if dezena in resultado_set else 0)

    X_ts = np.array(X_ts_list, dtype=np.float32)
    X_feat = np.array(X_feat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Usa o scaler JÁ APRENDIDO no treino forte (/treinar)
    X_feat_scaled = (X_feat - wrapper.mean_feat) / wrapper.std_feat

    # ------------------------------------------------
    # HIPERPARÂMETROS DE TREINO INCREMENTAL
    # ------------------------------------------------
    if INTENSIVE_LEARNING:
        # Incremental forte, mas ainda muito menor que /treinar
        epocas = 40
        batch = 256
    else:
        epocas = 8
        batch = 100

    net.fit(X_ts, X_feat_scaled, y, epochs=epocas, batch_size=batch)

    # ------------------------------------------------
    # APLICA PENALIDADES APÓS O TREINO
    # ------------------------------------------------
    aplicar_penalidades_apos_resultado(resultado_set, modo)

    # Salva e atualiza cache
    save_model(wrapper)
    global _model_cache
    _model_cache = wrapper

    logger.info(
        "Treino incremental pós-concurso CONCLUÍDO (modo %s, épocas=%d, amostras=%d) + penalidades aplicadas.",
        "INTENSIVO" if INTENSIVE_LEARNING else "NORMAL",
        epocas,
        len(y),
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
    seed = int(time.time() * 1000) % 1000000
    rng = np.random.default_rng(seed)
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
# FUNÇÃO DE REFINO ULTRA-EFICIENTE (VERSÃO CORRIGIDA)
# ----------------------------------------------------

def corrigir_repeticoes_entre_apostas(apostas: List[List[int]]) -> List[List[int]]:
    """
    Corrige repetições entre apostas garantindo que cada dezena apareça no máximo uma vez.
    """
    todas_dezenas = []
    for aposta in apostas:
        todas_dezenas.extend(aposta)
    
    contador = Counter(todas_dezenas)
    repetidas = {dezena: count for dezena, count in contador.items() if count > 1}
    
    if not repetidas:
        return apostas
    
    logger.info(f"Corrigindo {len(repetidas)} dezenas repetidas...")
    
    # Para cada dezena repetida, mantém em apenas uma aposta
    dezenas_processadas = set()
    apostas_corrigidas = []
    
    for ap_idx, aposta in enumerate(apostas):
        aposta_corrigida = []
        for d in aposta:
            if d not in dezenas_processadas:
                aposta_corrigida.append(d)
                dezenas_processadas.add(d)
        
        # Se perdeu dezenas, completa com dezenas não usadas
        if len(aposta_corrigida) < 50:
            dezenas_disponiveis = [d for d in range(100) if d not in dezenas_processadas]
            precisamos = 50 - len(aposta_corrigida)
            
            if precisamos <= len(dezenas_disponiveis):
                # Pega as dezenas com base na ordem original da aposta
                completar = dezenas_disponiveis[:precisamos]
                aposta_corrigida.extend(completar)
                dezenas_processadas.update(completar)
            else:
                # Emergência: usa qualquer dezena não repetida
                for d in range(100):
                    if len(aposta_corrigida) >= 50:
                        break
                    if d not in dezenas_processadas:
                        aposta_corrigida.append(d)
                        dezenas_processadas.add(d)
        
        apostas_corrigidas.append(sorted(aposta_corrigida))
    
    # Verificação final
    todas = []
    for ap in apostas_corrigidas:
        todas.extend(ap)
    
    if len(set(todas)) != 150:
        logger.error("Falha ao corrigir repetições!")
        # Último recurso: forçar 150 dezenas únicas
        return gerar_apostas_unicas_forcado()
    
    return apostas_corrigidas


def gerar_apostas_unicas_forcado() -> List[List[int]]:
    """
    Gera 3 apostas com 150 dezenas únicas (último recurso).
    """
    logger.warning("Usando geração forçada de apostas únicas...")
    todas_dezenas = list(range(100))
    rng = np.random.default_rng(int(time.time() * 1000) % 1000000)
    rng.shuffle(todas_dezenas)
    
    apostas = []
    for i in range(3):
        inicio = i * 50
        fim = inicio + 50
        apostas.append(sorted(todas_dezenas[inicio:fim]))
    
    return apostas


def selecionar_dezenas_refino_corrigido(
    score: np.ndarray, 
    dezenas_proibidas: Set[int] = None,
    n_dezenas: int = 50
) -> List[int]:
    """Seleção otimizada para /refino - VERSÃO CORRIGIDA."""
    score_temp = score.copy()
    
    # Penaliza dezenas proibidas
    if dezenas_proibidas:
        for d in dezenas_proibidas:
            score_temp[d] = -1000.0  # Penalidade EXTREMA
    
    # Ordena por score
    idx_ordenados = np.argsort(-score_temp)
    
    # Seleciona garantindo diversidade
    selecionadas = []
    for d in idx_ordenados:
        if len(selecionadas) >= n_dezenas:
            break
        
        # Verifica se esta dezena é muito similar às já selecionadas
        # (evita clusters)
        if len(selecionadas) >= 10:
            # Calcula "distância" média
            distancias = [abs(d - s) for s in selecionadas[-10:]]
            if min(distancias) < 3:  # Muito próximo
                continue
        
        # Verifica se a dezena já está selecionada (proteção extra)
        if int(d) in selecionadas:
            continue
            
        selecionadas.append(int(d))
    
    # Se não conseguiu 50, completa com as melhores disponíveis
    if len(selecionadas) < n_dezenas:
        logger.warning(f"Apenas {len(selecionadas)} dezenas selecionadas, completando...")
        for d in idx_ordenados:
            if len(selecionadas) >= n_dezenas:
                break
            d_int = int(d)
            if d_int not in selecionadas:
                # Verifica se não está próxima demais das já selecionadas
                if len(selecionadas) >= 10:
                    distancias = [abs(d_int - s) for s in selecionadas[-10:]]
                    if min(distancias) < 2:  # Mais rigoroso
                        continue
                selecionadas.append(d_int)
    
    # Verificação final
    if len(selecionadas) < n_dezenas:
        logger.error(f"CRÍTICO: Só conseguiu {len(selecionadas)} dezenas!")
        # Completa com qualquer dezena não usada
        for d in range(100):
            if len(selecionadas) >= n_dezenas:
                break
            if d not in selecionadas:
                selecionadas.append(d)
    
    return sorted(selecionadas)


def gerar_apostas_refino(
    history: List[Set[int]], model: ModelWrapper
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    COMANDO /refino - VERSÃO ULTRA-EFICIENTE REFEITA

    Ideia central:
    - Usa histórico + penalidades (aprendizado) para ranquear as dezenas
    - Cria um NÚCLEO de dezenas principais (fixas em todas as apostas)
    - Cria dezenas secundárias que GIRAM entre as 3 apostas
    - Cada /refino gera variações, mas mantém a espinha dorsal

    Retorna: (apostas, espelhos)
    """
    if len(history) < 5:
        raise ValueError("Histórico insuficiente para gerar apostas de refino.")

    logger.info("=== INICIANDO /REFINO (NOVA LÓGICA) ===")

    # ----------------------------------------------------
    # 1) ANÁLISE BÁSICA DO HISTÓRICO E PENALIDADES
    # ----------------------------------------------------
    n_hist = len(history)

    # Atrasos
    atrasos = np.zeros(100, dtype=np.int32)
    for d in range(100):
        gap = 0
        for conc in reversed(history):
            if d in conc:
                break
            gap += 1
        atrasos[d] = gap

    # Frequências em janelas
    janela_curta = min(8, n_hist)
    janela_media = min(20, n_hist)
    janela_longa = min(50, n_hist)

    freq_curta = np.zeros(100, dtype=np.int32)
    freq_media = np.zeros(100, dtype=np.int32)
    freq_longa = np.zeros(100, dtype=np.int32)

    for i, conc in enumerate(reversed(history)):
        if i < janela_curta:
            for d in conc:
                freq_curta[d] += 1
        if i < janela_media:
            for d in conc:
                freq_media[d] += 1
        if i < janela_longa:
            for d in conc:
                freq_longa[d] += 1

    # Probabilidades do modelo
    probs = gerar_probabilidades_para_proximo(history, model)
    probs = np.clip(probs, 1e-9, None)

    # Penalidades aprendidas
    penalidades = carregar_penalidades()
    penal_array = np.zeros(100, dtype=np.float32)
    for d, v in penalidades.items():
        penal_array[d] = float(v)

    # Dezenas perigosas e em sequência (últimos concursos)
    ultimos_5 = history[-5:] if len(history) >= 5 else history
    freq_ultimos_5 = np.zeros(100, dtype=np.int32)
    for conc in ultimos_5:
        for d in conc:
            freq_ultimos_5[d] += 1

    dezenas_perigosas = {d for d in range(100) if freq_ultimos_5[d] >= 3}

    dezenas_em_sequencia = set()
    if len(history) >= 2:
        ultimo = history[-1]
        penultimo = history[-2]
        dezenas_em_sequencia = ultimo.intersection(penultimo)

    logger.info(f"Dezenas perigosas: {sorted(dezenas_perigosas)}")
    logger.info(f"Dezenas em sequência: {sorted(dezenas_em_sequencia)}")

    # ----------------------------------------------------
    # 2) NORMALIZAÇÃO
    # ----------------------------------------------------
    def _norm(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        maxv = float(arr.max())
        if maxv <= 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        return arr / maxv

    atraso_n = _norm(atrasos)
    freq_curta_n = _norm(freq_curta)
    freq_media_n = _norm(freq_media)
    freq_longa_n = _norm(freq_longa)
    probs_n = _norm(probs)
    penal_n = _norm(penal_array)

    # ----------------------------------------------------
    # 3) SCORE GLOBAL (APRENDIZADO REAL)
    # ----------------------------------------------------
    # Score alto = dezena "interessante" para aparecer nas apostas de refino.
    # Ajuste de pesos pode ser afinado depois dos resultados reais.
    base_score = (
        (1.0 - penal_n) * 0.35  # menos penalizada pelo aprendizado
        + atraso_n * 0.25       # mais atrasada
        + (1.0 - freq_curta_n) * 0.15  # pouco usada recentemente
        + (1.0 - freq_media_n) * 0.10  # pouco usada na janela média
        + (1.0 - probs_n) * 0.15       # baixa probabilidade da rede
    )

    # Dezenas perigosas e em sequência recebem queda de score
    for d in dezenas_perigosas:
        base_score[d] -= 0.5
    for d in dezenas_em_sequencia:
        base_score[d] -= 0.3

    # Ranking global
    ranking = list(np.argsort(-base_score))  # ordem decrescente

    # ----------------------------------------------------
    # 4) DEFINIÇÃO DO NÚCLEO E DAS FLEX
    # ----------------------------------------------------
    # Núcleo fixo: sempre presente nas 3 apostas
    # Núcleo variável: gira entre as 3 apostas de forma diferente
    # Flex: completa para 50 dezenas, girando mais agressivamente

    seed = int(time.time() * 1000) % 1000000
    rng = np.random.default_rng(seed)
    logger.info(f"Seed refino (núcleo + flex): {seed}")

    # quantidades podem ser afinadas depois
    Q_FIXAS = 15
    Q_VARIAVEIS = 15  # serão usadas como subgrupos em cada aposta
    Q_TOTAL_CORE = Q_FIXAS + Q_VARIAVEIS  # 30 dezenas de "núcleo ampliado"

    core_fixas = ranking[:Q_FIXAS]
    core_var = ranking[Q_FIXAS:Q_TOTAL_CORE]

    # pool flex (evita perigosas e sequências pesadas)
    pool_flex = [
        d for d in ranking[Q_TOTAL_CORE:]
        if d not in dezenas_perigosas
    ]
    if len(pool_flex) < 60:
        # fallback: permite algumas perigosas para não travar
        extra = [d for d in ranking[Q_TOTAL_CORE:] if d not in pool_flex]
        pool_flex.extend(extra)

    logger.info(f"Núcleo fixo: {sorted(core_fixas)}")
    logger.info(f"Núcleo variável (30-15): {sorted(core_var)}")
    logger.info(f"Tamanho pool flex: {len(pool_flex)}")

    apostas_refino: List[List[int]] = []
    flex_usadas_global: Set[int] = set()

    # ----------------------------------------------------
    # 5) CONSTRUÇÃO DAS 3 APOSTAS
    # ----------------------------------------------------
    for idx in range(3):
        # começa com o núcleo fixo
        numeros = set(int(d) for d in core_fixas)

        # escolhe parte do núcleo variável de forma diferente em cada aposta
        core_var_lista = core_var.copy()
        if idx == 0:
            # primeira aposta: core_var como está
            pass
        elif idx == 1:
            core_var_lista = list(reversed(core_var_lista))
        else:
            core_var_lista = list(rng.permutation(core_var_lista))

        for d in core_var_lista:
            if len(numeros) >= 30:  # 15 fixas + 15 variáveis
                break
            numeros.add(int(d))

        # agora precisamos completar até 50 com dezenas flex
        # score flex parte do base_score, mas:
        score_flex = base_score.copy()

        # penaliza forte as perigosas e em sequência
        for d in dezenas_perigosas:
            score_flex[d] -= 1.0
        for d in dezenas_em_sequencia:
            score_flex[d] -= 0.7

        # penaliza moderadamente os flex já usados em outras apostas
        for d in flex_usadas_global:
            score_flex[d] -= 0.4

        # gera barulho controlado para girar entre refinos
        ruido = rng.normal(0, 0.05, size=100)
        score_flex = score_flex + ruido

        candidatos = [d for d in pool_flex if d not in numeros]
        candidatos_ordenados = sorted(
            candidatos,
            key=lambda d: score_flex[d],
            reverse=True,
        )

        for d in candidatos_ordenados:
            if len(numeros) >= 50:
                break
            numeros.add(int(d))

        # se por algum motivo ainda não chegou em 50, completa com o ranking global
        if len(numeros) < 50:
            for d in ranking:
                if len(numeros) >= 50:
                    break
                if d not in numeros:
                    numeros.add(int(d))

        aposta = sorted(numeros)
        apostas_refino.append(aposta)

        # registra flex usados (tudo que não é núcleo fixo)
        flex_usadas_global.update(set(aposta) - set(core_fixas))

        logger.info(f"Aposta {idx+1} gerada com {len(aposta)} dezenas.")

    # ----------------------------------------------------
    # 6) ESPELHOS (APENAS COMO COMPLEMENTO)
    # ----------------------------------------------------
    universo = set(range(100))
    espelhos_refino: List[List[int]] = []
    for aposta in apostas_refino:
        espelho = sorted(list(universo - set(aposta)))
        espelhos_refino.append(espelho)

    # ----------------------------------------------------
    # 7) LOG DE QUALIDADE E DIVERSIDADE
    # ----------------------------------------------------
    logger.info("=== ANÁLISE FINAL DO /REFINO (NOVA LÓGICA) ===")
    for i, aposta in enumerate(apostas_refino, 1):
        perigosas_na_aposta = len(set(aposta) & dezenas_perigosas)
        sequencia_na_aposta = len(set(aposta) & dezenas_em_sequencia)
        logger.info(
            f"Aposta {i}: {len(aposta)} dezenas, "
            f"{perigosas_na_aposta} perigosas, {sequencia_na_aposta} em sequência"
        )

    # diversidade entre apostas (permitida, mas monitorada)
    sets_ap = [set(ap) for ap in apostas_refino]
    over_12 = len(sets_ap[0].intersection(sets_ap[1]))
    over_13 = len(sets_ap[0].intersection(sets_ap[2]))
    over_23 = len(sets_ap[1].intersection(sets_ap[2]))
    logger.info(
        f"Sobreposição entre apostas (qtd dezenas iguais): "
        f"1x2={over_12}, 1x3={over_13}, 2x3={over_23}"
    )

    todas = []
    for ap in apostas_refino:
        todas.extend(ap)
    unique_total = len(set(todas))
    logger.info(f"Total de dezenas distintas no bloco: {unique_total} (máx=100)")

    logger.info(f"Refino (nova lógica) gerado com sucesso (seed: {seed})")

    return apostas_refino, espelhos_refino


# ----------------------------------------------------
# HANDLERS TELEGRAM
# ----------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🔮 Oráculo Lotomania – Modo C (Híbrido CNN + MLP – Intensivo)\n\n"
        "/treinar - treina ou atualiza a rede neural híbrida (treino forte) "
        "(RESTRITO à whitelist)\n"
        "/gerar - Oráculo Supremo (6 apostas + 6 espelhos)\n"
        "/bolao - 🧠 NOVO: Gerador estratégico (4 apostas de 50 dezenas, janela 50, "
        "núcleo + funções definidas)\n"
        "/refino - 🎯 Versão ULTRA-EFICIENTE focada em penalidades e variações\n"
        "/confirmar - confronta o resultado oficial com o último bloco gerado, "
        "registra desempenho e aplica treino incremental + penalidades "
        "(RESTRITO à whitelist)\n"
        "/avaliar - apenas confirma os acertos das apostas (SEM treinar o modelo)\n"
        "/status_penalidades - mostra as dezenas mais penalizadas/recompensadas\n\n"
        "🎯 DIFERENÇAS ENTRE MODOS:\n"
        "• /gerar: Versão tradicional orientada a acertos diretos\n"
        "• /bolao: Estratégia estatística com núcleo fixo e rotações inteligentes\n"
        "• /refino: Exploração agressiva com penalidades exponenciais e espelhos\n\n"
        f"Modo treino habilitado: {'SIM' if TREINO_HABILITADO else 'NÃO (apenas avaliação)'}\n"
        f"Modo intensivo: {'ATIVADO' if INTENSIVE_LEARNING else 'DESATIVADO'}\n"
        f"Sistema de penalidades: ATIVADO (P={PENALIDADE_ERRO}, R={RECOMPENSA_ACERTO_ERRAR})"
    )
    await update.message.reply_text(msg)


async def status_penalidades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Mostra o status das penalidades acumuladas.
    """
    penalidades = carregar_penalidades()
    
    if not penalidades:
        await update.message.reply_text("📊 Nenhuma penalidade registrada ainda.")
        return
    
    # Separa penalidades positivas (dezenas a evitar) e negativas (dezenas a preferir)
    positivas = {d: v for d, v in penalidades.items() if v > 0}
    negativas = {d: v for d, v in penalidades.items() if v < 0}
    
    linhas = ["📊 STATUS DAS PENALIDADES"]
    linhas.append("")
    
    if positivas:
        linhas.append("🔴 DEZENAS A EVITAR (penalidades positivas):")
        positivas_ordenadas = sorted(positivas.items(), key=lambda x: x[1], reverse=True)[:20]
        for dezena, valor in positivas_ordenadas:
            linhas.append(f"  {dezena:02d}: +{valor:.3f}")
        linhas.append("")
    
    if negativas:
        linhas.append("🟢 DEZENAS A PREFERIR (penalidades negativas):")
        negativas_ordenadas = sorted(negativas.items(), key=lambda x: x[1])[:20]
        for dezena, valor in negativas_ordenadas:
            linhas.append(f"  {dezena:02d}: {valor:.3f}")
        linhas.append("")
    
    linhas.append(f"Total de dezenas com penalidades: {len(penalidades)}")
    linhas.append(f"Penalidade máxima configurada: {MAX_PENALIDADE}")
    linhas.append(f"Decaimento por concurso: {(1-DECAIMENTO_PENALIDADE)*100:.1f}%")
    
    await update.message.reply_text("\n".join(linhas))


async def avaliar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /avaliar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94
    
    Apenas confirma os acertos das apostas SEM treinar o modelo.
    Ideal para testar estratégias sem afetar o aprendizado.
    """
    texto = (update.message.text or "").strip()
    partes = texto.split()

    if len(partes) < 21:
        await update.message.reply_text(
            "❌ Uso correto:\n"
            "/avaliar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94"
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
            "Gere um novo bloco com /gerar ou /refino e depois use /avaliar."
        )
        return

    try:
        with open(ULTIMA_GERACAO_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)

        apostas = dados.get("apostas")
        espelhos = dados.get("espelhos")
        modo = dados.get("modo", "oraculo")  # "oraculo" ou "refino"

        if not apostas or not espelhos:
            raise ValueError("Dados incompletos na última geração.")

        # garante int nativo
        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

    except Exception:
        logger.exception("Erro ao ler arquivo de última geração.")
        await update.message.reply_text(
            "⚠️ Arquivo de última geração está corrompido ou em formato antido.\n"
            "Use /gerar ou /refino novamente para criar um novo bloco."
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
    # 4) Resposta para o usuário
    # ----------------------------------
    linhas = []
    linhas.append("📊 Avaliação de Desempenho (SEM treino)")
    linhas.append("Dezenas sorteadas:")
    linhas.append(" ".join(f"{d:02d}" for d in sorted(resultado)))
    linhas.append("")
    linhas.append(f"Melhor aposta do lote: {melhor_hits} acertos")
    linhas.append(f"Média de acertos do lote: {media_hits:.2f}")
    linhas.append("")

    if modo == "refino":
        labels = [
            "Aposta 1 – Evitação Radical",
            "Aposta 2 – Híbrida Otimizada", 
            "Aposta 3 – Dezenas Seguras"
        ]
        labels = labels[:n_apostas]
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

    if modo == "refino":
        linhas.append(
            f"🏅 Melhor aposta refino: {labels[melhor_ap_idx]} ({hits_apostas[melhor_ap_idx]} pontos)"
        )
        linhas.append(
            f"🏅 Melhor espelho refino: Espelho {melhor_esp_idx+1} ({hits_espelhos[melhor_esp_idx]} pontos)"
        )
    else:
        linhas.append(
            f"🏅 Melhor aposta: {labels[melhor_ap_idx]} ({hits_apostas[melhor_ap_idx]} pontos)"
        )
        linhas.append(
            f"🏅 Melhor espelho: Espelho {melhor_esp_idx+1} ({hits_espelhos[melhor_esp_idx]} pontos)"
        )

    linhas.append("\nℹ️ Este comando apenas avalia os acertos, SEM alterar o modelo.")
    linhas.append("Use /confirmar para treinar o modelo com este resultado.")

    await update.message.reply_text("\n".join(linhas).strip())


async def confirmar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /confirmar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94
    """
    if not is_user_whitelisted(update):
        usuario = get_user_label(update)
        await update.message.reply_text(
            f"⚠️ {usuario}, você não tem permissão para usar /confirmar."
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

    # Parse do resultado
    try:
        dezenas_str = partes[1:]
        dezenas_int = [int(d) for d in dezenas_str]
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
            "❌ Não consegui interpretar as dezenas."
        )
        return

    # Carrega última geração
    if not os.path.exists(ULTIMA_GERACAO_PATH):
        await update.message.reply_text(
            "⚠️ Arquivo de última geração não encontrado."
        )
        return

    try:
        with open(ULTIMA_GERACAO_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)

        apostas = dados.get("apostas")
        espelhos = dados.get("espelhos")
        modo = dados.get("modo", "oraculo")
        user_id_gerador = dados.get("user_id")

        if not apostas or not espelhos:
            raise ValueError("Dados incompletos.")

        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

    except Exception:
        logger.exception("Erro ao ler arquivo de última geração.")
        await update.message.reply_text(
            "⚠️ Arquivo de última geração corrompido."
        )
        return

    # Verifica usuário
    current_user = update.effective_user
    current_id = current_user.id if current_user else None

    if user_id_gerador is not None and current_id is not None and current_id != user_id_gerador:
        await update.message.reply_text(
            "⚠️ O último bloco foi gerado por outro usuário."
        )
        return

    # Calcula acertos
    hits_apostas = []
    hits_espelhos = []

    for ap, esp in zip(apostas_py, espelhos_py):
        hits_apostas.append(len(resultado_set.intersection(ap)))
        hits_espelhos.append(len(resultado_set.intersection(esp)))

    n_apostas = len(hits_apostas)
    
    if n_apostas == 0:
        await update.message.reply_text("⚠️ Não há apostas válidas.")
        return

    melhor_ap_idx = int(np.argmax(hits_apostas))
    melhor_esp_idx = int(np.argmax(hits_espelhos))

    melhor_hits = int(hits_apostas[melhor_ap_idx])
    media_hits = float(sum(hits_apostas) / n_apostas)

    # Classificação do lote
    if melhor_hits >= 15:
        classe_lote = "Lote EXCELENTE"
        epochs_inc = 20
    elif melhor_hits >= 11:
        classe_lote = "Lote forte"
        epochs_inc = 28
    elif melhor_hits >= 9:
        classe_lote = "Lote mediano"
        epochs_inc = 22
    else:
        classe_lote = "Lote fraco"
        epochs_inc = 16

    # Salva histórico
    try:
        existe = os.path.exists(DESEMPENHO_PATH)
        with open(DESEMPENHO_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")

            if not existe:
                header = [
                    "timestamp", "resultado",
                    "acertos_ap1", "acertos_ap2", "acertos_ap3",
                    "acertos_ap4", "acertos_ap5", "acertos_ap6",
                    "acertos_esp1", "acertos_esp2", "acertos_esp3",
                    "acertos_esp4", "acertos_esp5", "acertos_esp6",
                    "melhor_aposta", "melhor_espelho",
                    "modo", "melhor_hits", "media_hits", "tipo_confirma",
                ]
                writer.writerow(header)

            ts = time.time()
            resultado_txt = " ".join(f"{d:02d}" for d in sorted(resultado))

            ha = (hits_apostas + [0] * 6)[:6]
            he = (hits_espelhos + [0] * 6)[:6]

            row = [
                f"{ts:.3f}", resultado_txt,
                *[int(h) for h in ha],
                *[int(h) for h in he],
                melhor_ap_idx + 1, melhor_esp_idx + 1,
                modo, melhor_hits, f"{media_hits:.2f}", "confirmar",
            ]
            writer.writerow(row)

        logger.info("Desempenho registrado em %s", DESEMPENHO_PATH)

    except Exception as e_csv:
        logger.exception("Erro ao salvar desempenho em CSV: %s", e_csv)

    # Treino incremental com penalidades
    if TREINO_HABILITADO:
        try:
            history = load_history(HISTORY_PATH)
            if history is not None:
                treino_incremental_pos_concurso(history, resultado_set, modo)
                txt_treino = (
                    f"\n🧠 Treino incremental INTENSIVO aplicado ao modelo (CNN+MLP).\n"
                    f"   • Melhor aposta do lote: {melhor_hits} acertos\n"
                    f"   • Média do lote: {media_hits:.2f} acertos\n"
                    f"   • Modo: {'INTENSIVO (80 épocas, janela 30 concursos)' if INTENSIVE_LEARNING else 'NORMAL'}\n"
                    f"   • Sistema de penalidades ATIVADO para modo '{modo}'"
                )
            else:
                txt_treino = "\n⚠️ Não foi possível carregar o histórico para treino."
        except Exception as e_inc:
            logger.exception("Erro no treino incremental: %s", e_inc)
            txt_treino = "\n⚠️ Não foi possível aplicar o treino incremental."
    else:
        txt_treino = "\nℹ️ Modo avaliação: SEM treino."

    # Atualiza snapshot do melhor modelo
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
                    "\n🏆 Este lote superou o melhor desempenho anterior."
                )
        except Exception as e_best:
            logger.exception("Erro ao registrar snapshot: %s", e_best)

    # Resposta
    linhas = []
    linhas.append("✅ Resultado confirmado!")
    linhas.append("Dezenas sorteadas:")
    linhas.append(" ".join(f"{d:02d}" for d in sorted(resultado)))
    linhas.append("")
    linhas.append(f"Melhor aposta do lote: {melhor_hits} acertos")
    linhas.append(f"Média de acertos do lote: {media_hits:.2f}")
    linhas.append(classe_lote)
    linhas.append("")

    if modo == "refino":
        labels = [
            "Aposta 1 – Evitação Radical",
            "Aposta 2 – Híbrida Otimizada", 
            "Aposta 3 – Dezenas Seguras"
        ]
        labels = labels[:n_apostas]
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

    if modo == "refino":
        linhas.append(
            f"🏅 Melhor aposta refino: {labels[melhor_ap_idx]} ({hits_apostas[melhor_ap_idx]} pontos)"
        )
        linhas.append(
            f"🏅 Melhor espelho refino: Espelho {melhor_esp_idx+1} ({hits_espelhos[melhor_esp_idx]} pontos)"
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
    """
    Treinamento completo do modelo híbrido CNN+MLP.
    """
    if not is_user_whitelisted(update):
        usuario = get_user_label(update)
        await update.message.reply_text(
            f"⚠️ {usuario}, você não tem permissão para usar /treinar."
        )
        return

    try:
        modo_txt = "INTENSIVO" if INTENSIVE_LEARNING else "normal"
        await update.message.reply_text(
            f"🧠 Iniciando treinamento híbrido (CNN + MLP) – modo {modo_txt}..."
        )

        history = load_history(HISTORY_PATH)
        seq_len_default = 60 if INTENSIVE_LEARNING else 40

        wrapper_antigo = None
        seq_len = min(seq_len_default, len(history) - 1)

        try:
            wrapper_antigo = load_model_local()
            seq_len = wrapper_antigo.seq_len
        except FileNotFoundError:
            wrapper_antigo = None
        except RuntimeError as e_incompat:
            logger.warning(str(e_incompat))
            wrapper_antigo = None
        except Exception as e:
            logger.exception("Erro ao carregar modelo antigo.")
            wrapper_antigo = None

        X_ts, X_feat, y = build_dataset_hybrid(history, seq_len)
        mean_feat, std_feat = compute_scaler(X_feat)
        X_feat_scaled = (X_feat - mean_feat) / std_feat

        feat_dim = X_feat.shape[1]

        if wrapper_antigo is not None:
            net = wrapper_antigo.net
        else:
            net = HybridCNNMLP(seq_len=seq_len, feat_dim=feat_dim)

        if INTENSIVE_LEARNING:
            epocas = 120
            batch = 512
        else:
            epocas = 60
            batch = 512

        net.fit(X_ts, X_feat_scaled, y, epochs=epocas, batch_size=batch)

        wrapper = ModelWrapper(net, mean_feat, std_feat, seq_len)
        save_model(wrapper)
        global _model_cache
        _model_cache = wrapper

        await update.message.reply_text(
            "✅ Treinamento concluído (híbrido CNN+MLP).\n"
            f"Amostras usadas: {len(y)}\n"
            f"seq_len = {seq_len}\n"
            f"épocas = {epocas} (modo {modo_txt})"
        )

    except Exception as e:
        logger.exception("Erro no treinamento")
        await update.message.reply_text(f"❌ Erro no treinamento: {e}")


async def gerar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        history = load_history(HISTORY_PATH)
        model = load_model_local()

        apostas, espelhos = gerar_apostas_oraculo_supremo(history, model)

        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

        try:
            user = update.effective_user
            dados = {
                "timestamp": float(time.time()),
                "modo": "oraculo",
                "apostas": apostas_py,
                "espelhos": espelhos_py,
                "user_id": user.id if user else None,
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


# ============================================================
# /bolao — NOVO COMANDO (sem mexer no /gerar)
# Janela: últimos 50 concursos
# Saída: 4 apostas (50 dezenas) + espelhos (compat /avaliar)
# Correção: VARIAÇÃO CONTROLADA por execução (seed variável + amostragem ponderada)
# ============================================================

def _bolao_window(history, janela: int = 50):
    if not history:
        raise ValueError("Histórico vazio.")
    return history[-janela:] if len(history) >= janela else history


def _count_freq_and_delay(hist_window):
    freq = np.zeros(100, dtype=np.int32)
    for conc in hist_window:
        for d in conc:
            freq[d] += 1

    delay = np.zeros(100, dtype=np.int32)
    for d in range(100):
        gap = 0
        found = False
        for conc in reversed(hist_window):
            if d in conc:
                found = True
                break
            gap += 1
        delay[d] = gap if found else len(hist_window) + 5

    return freq, delay


def _cooc_pairs(hist_window):
    cooc = np.zeros((100, 100), dtype=np.int16)
    for conc in hist_window:
        arr = sorted(conc)
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i + 1, len(arr)):
                b = arr[j]
                cooc[a, b] += 1
                cooc[b, a] += 1
    return cooc


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _choose_top(scores: np.ndarray, k: int, forbid: set[int] | None = None) -> list[int]:
    forbid = forbid or set()
    idxs = [i for i in range(100) if i not in forbid]
    idxs.sort(key=lambda i: float(scores[i]), reverse=True)
    return idxs[:k]


def _weighted_sample_no_replace(scores: np.ndarray, k: int, rng, forbid: set[int] | None = None, temperature: float = 1.0):
    """
    Amostragem ponderada SEM reposição.
    - scores: quanto maior, mais chance
    - temperature: <1.0 deixa mais "top"; >1.0 deixa mais "solto"
    """
    forbid = forbid or set()
    idxs = np.array([i for i in range(100) if i not in forbid], dtype=np.int32)
    if idxs.size == 0:
        return []

    s = scores[idxs].astype(np.float64)

    # estabiliza e garante positividade
    s = s - s.min()
    s = np.clip(s, 0.0, None) + 1e-9

    # temperatura (controla o quanto varia)
    if temperature <= 0:
        temperature = 1.0
    s = np.power(s, 1.0 / temperature)

    p = s / s.sum()

    k = int(min(k, idxs.size))
    chosen = rng.choice(idxs, size=k, replace=False, p=p)
    return [int(x) for x in chosen]


def _stratified_pick(candidates: list[int], k: int, rng, bands=None) -> list[int]:
    if bands is None:
        bands = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)]

    cand_set = set(candidates)
    buckets = []
    for a, b in bands:
        buckets.append([d for d in range(a, b + 1) if d in cand_set])

    target = [k // len(buckets)] * len(buckets)
    rem = k - sum(target)
    i = 0
    while rem > 0:
        target[i] += 1
        rem -= 1
        i = (i + 1) % len(target)

    chosen = []
    for bucket, t in zip(buckets, target):
        if not bucket:
            continue
        rng.shuffle(bucket)
        chosen.extend(bucket[:t])

    if len(chosen) < k:
        rest = [d for d in candidates if d not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: (k - len(chosen))])

    return sorted(chosen[:k])


def gerar_apostas_bolao(history: list[set[int]], rng, bolao_run: int) -> tuple[list[list[int]], list[list[int]]]:
    """
    Gera 4 apostas (50 dezenas) com funções:
      A1: Quentes + Puxadas (tendência)
      A2: Alternância com viés alto (60–99)
      A3: Atrasadas/retorno (quebra de ciclo)
      A4: Balanceada robusta (estratificada)
    Retorna (apostas, espelhos)

    ✅ Agora varia a cada execução:
    - seed variável (passado via rng)
    - seleção ponderada na periferia (mantém lógica, mas gira dezenas)
    - micro-núcleo estável e controlado
    """
    hist_window = _bolao_window(history, 50)
    freq, delay = _count_freq_and_delay(hist_window)
    cooc = _cooc_pairs(hist_window)

    freq_n = _norm01(freq)
    delay_n = _norm01(delay)
    recency_n = 1.0 - delay_n
    delay_hi = _norm01(delay)  # maior = mais atrasada

    # -----------------------------
    # Micro-núcleo (4 dezenas) — ESTÁVEL (sem rng)
    # -----------------------------
    score_nucleo = (0.55 * freq_n) + (0.45 * recency_n) - (0.08 * (freq_n ** 2))
    nucleo = _choose_top(score_nucleo, 4)
    nucleo_set = set(nucleo)

    # Pequena rotação controlada do núcleo: 1 dezena pode girar a cada 3 execuções
    # (mantém identidade, mas evita "congelar" pra sempre)
    if bolao_run % 3 == 0:
        cand_nucleo = _choose_top(score_nucleo, 10, forbid=set(nucleo))
        if cand_nucleo:
            troca_out = nucleo[-1]
            troca_in = cand_nucleo[0]
            nucleo = sorted(list((set(nucleo) - {troca_out}) | {troca_in}))
            nucleo_set = set(nucleo)

    # -----------------------------
    # Helpers de puxada
    # -----------------------------
    def pull_from(set_base: set[int]) -> np.ndarray:
        ps = np.zeros(100, dtype=np.float32)
        base_list = list(set_base)
        for d in range(100):
            if d in set_base:
                ps[d] = 0.0
            else:
                ps[d] = float(np.sum(cooc[d, base_list])) if base_list else 0.0
        return _norm01(ps)

    # -----------------------------
    # A1 — Quentes + Puxadas (variação leve)
    # -----------------------------
    hot = set(_choose_top(freq_n, 18)) | nucleo_set
    pull_n = pull_from(hot)

    score_a1 = (0.60 * freq_n) + (0.30 * pull_n) + (0.10 * recency_n)

    # fixa uma base “core” e gira o resto
    core_a1 = set(_choose_top(score_a1, 28)) | nucleo_set  # 28 mais fortes
    forbid_a1 = set(core_a1)
    # gira 22 restantes por amostragem ponderada
    extra_a1 = _weighted_sample_no_replace(score_a1, 50 - len(core_a1), rng, forbid=forbid_a1, temperature=0.85)
    base_a1 = sorted(list(core_a1 | set(extra_a1)))

    # -----------------------------
    # A2 — Alternância (viés alto) com rotação real
    # -----------------------------
    band_bias = np.zeros(100, dtype=np.float32)
    for d in range(100):
        if d >= 80:
            band_bias[d] = 1.0
        elif d >= 60:
            band_bias[d] = 0.65
        elif d >= 40:
            band_bias[d] = 0.25
        else:
            band_bias[d] = 0.10

    pull2_n = pull_from(nucleo_set)
    score_a2 = (0.45 * freq_n) + (0.25 * pull2_n) + (0.10 * recency_n) + (0.20 * band_bias)

    core_a2 = set(_choose_top(score_a2, 24)) | nucleo_set
    forbid_a2 = set(core_a2)
    extra_a2 = _weighted_sample_no_replace(score_a2, 50 - len(core_a2), rng, forbid=forbid_a2, temperature=0.95)
    base_a2 = sorted(list(core_a2 | set(extra_a2)))

    # -----------------------------
    # A3 — Atrasadas/retorno (quebra) com rotação real
    # -----------------------------
    score_a3 = (0.65 * delay_hi) + (0.20 * (1.0 - freq_n)) + (0.15 * pull2_n)
    forbid_hot = set(_choose_top(freq_n, 10))
    # core da quebra (mais atrasadas que não sejam super quentes)
    core_candidates_a3 = [d for d in _choose_top(score_a3, 40) if d not in forbid_hot]
    core_a3 = set(core_candidates_a3[:22]) | nucleo_set
    forbid_a3 = set(core_a3) | forbid_hot
    extra_a3 = _weighted_sample_no_replace(score_a3, 50 - len(core_a3), rng, forbid=forbid_a3, temperature=1.05)
    base_a3 = sorted(list(core_a3 | set(extra_a3)))

    # -----------------------------
    # A4 — Balanceada robusta (estratificada) + rotação
    # -----------------------------
    score_a4 = (0.45 * freq_n) + (0.25 * recency_n) + (0.30 * pull2_n)
    # pega um pool grande via amostragem ponderada e depois estratifica
    pool_a4 = set(_choose_top(score_a4, 30)) | set(_weighted_sample_no_replace(score_a4, 60, rng, forbid=set(), temperature=0.95))
    pool_a4 = sorted(list(pool_a4 | nucleo_set))
    base_a4 = _stratified_pick(pool_a4, 50, rng=rng)
    base_a4 = sorted(set(base_a4) | nucleo_set)
    if len(base_a4) > 50:
        rest = [d for d in base_a4 if d not in nucleo_set]
        rest.sort(key=lambda d: float(score_a4[d]), reverse=True)
        base_a4 = sorted(list(nucleo_set) + rest[: (50 - len(nucleo_set))])

    # Garantia final de tamanho 50
    def _ensure50(ap: list[int], score: np.ndarray):
        s = set(ap)
        if len(s) > 50:
            # corta piores mantendo nucleo
            keep = set(nucleo)
            rest = [d for d in s if d not in keep]
            rest.sort(key=lambda d: float(score[d]), reverse=True)
            return sorted(list(keep) + rest[: (50 - len(keep))])
        if len(s) < 50:
            fill = [d for d in range(100) if d not in s]
            fill.sort(key=lambda d: float(score[d]), reverse=True)
            s.update(fill[: (50 - len(s))])
        return sorted(list(s))

    base_a1 = _ensure50(base_a1, score_a1)
    base_a2 = _ensure50(base_a2, score_a2)
    base_a3 = _ensure50(base_a3, score_a3)
    base_a4 = _ensure50(base_a4, score_a4)

    apostas = [
        [int(x) for x in base_a1],
        [int(x) for x in base_a2],
        [int(x) for x in base_a3],
        [int(x) for x in base_a4],
    ]

    universo = set(range(100))
    espelhos = [sorted(list(universo - set(ap))) for ap in apostas]

    return apostas, espelhos


async def bolao_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /bolao
    Gera 4 apostas (50 dezenas) com funções definidas (janela 50).
    Compatível com /avaliar e /confirmar porque salva também espelhos.
    ✅ Agora varia a cada execução (seed variável + contador persistido).
    """
    try:
        history = load_history(HISTORY_PATH)

        # Lê último estado (contador) do ULTIMA_GERACAO_PATH, se existir
        bolao_run = 0
        try:
            if os.path.exists(ULTIMA_GERACAO_PATH):
                with open(ULTIMA_GERACAO_PATH, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if isinstance(prev, dict) and prev.get("modo") == "bolao":
                    bolao_run = int(prev.get("bolao_run", 0))
        except Exception:
            bolao_run = 0

        bolao_run += 1

        user = update.effective_user
        user_id = int(user.id) if user else 0

        # Seed variável REAL (tempo + user + contador) -> muda toda execução
        seed = (int(time.time() * 1000) ^ (user_id * 1315423911) ^ (bolao_run * 97531)) % 1_000_000
        rng = np.random.default_rng(seed)

        apostas, espelhos = gerar_apostas_bolao(history, rng=rng, bolao_run=bolao_run)

        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

        # salva última geração (compat com /avaliar)
        try:
            dados = {
                "timestamp": float(time.time()),
                "modo": "bolao",
                "bolao_run": bolao_run,
                "seed": int(seed),
                "apostas": apostas_py,
                "espelhos": espelhos_py,
                "user_id": user_id,
            }
            with open(ULTIMA_GERACAO_PATH, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
        except Exception as e_save:
            logger.exception(f"Erro ao salvar última geração (/bolao): {e_save}")

        def fmt(lista):
            return " ".join(f"{d:02d}" for d in sorted(lista))

        linhas = []
        linhas.append(f"🧠 /BOLAO — 4 APOSTAS (Janela 50) — Funções definidas | run={bolao_run} | seed={seed}")
        linhas.append("")

        labels = [
            "✅ Aposta 1 — Quentes + Puxadas",
            "✅ Aposta 2 — Alternância (viés alto)",
            "✅ Aposta 3 — Retorno/Atrasadas (quebra)",
            "✅ Aposta 4 — Balanceada robusta",
        ]

        for i, (ap, esp) in enumerate(zip(apostas_py, espelhos_py), start=1):
            linhas.append(labels[i - 1])
            linhas.append(fmt(ap))
            linhas.append(f"Espelho {i}:")
            linhas.append(fmt(esp))
            linhas.append("")

        await update.message.reply_text("\n".join(linhas).strip())

    except Exception as e:
        logger.exception("Erro no comando /bolao")
        await update.message.reply_text(f"⚠️ Erro no /bolao: {e}")


async def refino_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    COMANDO /refino - VERSÃO ULTRA-EFICIENTE

    Gera 3 apostas usando TUDO que aprendemos:
    - Penalidades exponenciais
    - Análise de sequências
    - Evitação inteligente de dezenas quentes
    - Otimização para espelhos
    - Diversificação garantida
    """
    try:
        history = load_history(HISTORY_PATH)
        model = load_model_local()

        # Usa a versão ULTRA-EFICIENTE (gerador principal)
        apostas_refino, espelhos_refino = gerar_apostas_refino(history, model)

        # CONVERTE para int nativo
        apostas_py = [[int(x) for x in ap] for ap in apostas_refino]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos_refino]

        # ====================================================
        # BLINDAGEM: garante que a APOSTA 3 venha com dezenas
        # ====================================================
        # Garante pelo menos 3 apostas/espelhos nas estruturas
        while len(apostas_py) < 3:
            apostas_py.append([])
        while len(espelhos_py) < 3:
            espelhos_py.append([])

        # Se a Aposta 3 veio vazia ou com menos de 50 dezenas,
        # reconstruímos uma Aposta 3 segura a partir das dezenas
        # que ainda não foram usadas nas Apostas 1 e 2.
        if len(apostas_py[2]) < 50:
            universo = list(range(100))

            usadas = set(apostas_py[0]) | set(apostas_py[1])
            # Dezenas ainda não usadas nas duas primeiras apostas
            restantes = [d for d in universo if d not in usadas]

            # Se ainda assim faltar dezena para chegar em 50,
            # completa reaproveitando algumas (sem travar).
            if len(restantes) < 50:
                for d in universo:
                    if len(restantes) >= 50:
                        break
                    if d not in restantes:
                        restantes.append(d)

            aposta3 = sorted(restantes[:50])
            apostas_py[2] = aposta3

            # Recalcula o espelho da Aposta 3 para manter coerência
            universo_set = set(universo)
            espelhos_py[2] = sorted(universo_set - set(aposta3))

        # Salva como "modo = refino" para o /confirmar e /avaliar
        try:
            user = update.effective_user
            dados = {
                "timestamp": float(time.time()),
                "modo": "refino",
                "apostas": apostas_py,
                "espelhos": espelhos_py,
                "user_id": user.id if user else None,
            }
            with open(ULTIMA_GERACAO_PATH, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
        except Exception as e_save:
            logger.exception(f"Erro ao salvar última geração (refino): {e_save}")

        # Formata resposta
        linhas = []
        linhas.append("🎯 COMANDO /REFINO - VERSÃO ULTRA-EFICIENTE")
        linhas.append("=" * 50)
        linhas.append("")

        linhas.append("📊 ESTRATÉGIAS APLICADAS:")
        linhas.append("1. Penalidades exponenciais para sequências")
        linhas.append("2. Blacklist automática para dezenas persistentes")
        linhas.append("3. Otimização para maximizar acertos nos ESPELHOS")
        linhas.append("4. Diversificação garantida entre as apostas")
        linhas.append("")

        # Aposta 1
        linhas.append("🔴 APOSTA 1 - EVITAÇÃO RADICAL:")
        linhas.append("• Foca em dezenas NÃO perigosas")
        linhas.append("• Penaliza extremamente sequências")
        linhas.append("• Prioriza dezenas frias")
        linhas.append("")
        linhas.append(format_dezenas_sortidas(apostas_py[0]))
        linhas.append("")

        # Aposta 2
        linhas.append("🟡 APOSTA 2 - HÍBRIDA OTIMIZADA:")
        linhas.append("• Combina múltiplos fatores")
        linhas.append("• Balanceamento inteligente")
        linhas.append("• Diversificação em relação à Aposta 1")
        linhas.append("")
        linhas.append(format_dezenas_sortidas(apostas_py[1]))
        linhas.append("")

        # Aposta 3
        linhas.append("🟢 APOSTA 3 - DEZENAS SEGURAS:")
        linhas.append("• Pool restrito de dezenas seguras")
        linhas.append("• Máxima diversificação")
        linhas.append("• Foco em estabilidade")
        linhas.append("")
        linhas.append(format_dezenas_sortidas(apostas_py[2]))
        linhas.append("")

        linhas.append("📈 ESPELHOS OTIMIZADOS PARA MÁXIMOS ACERTOS")
        linhas.append("")
        linhas.append("💡 DICA: Use /avaliar para testar sem treinar")
        linhas.append("       Use /confirmar para aplicar aprendizado")

        await update.message.reply_text("\n".join(linhas))

    except Exception as e:
        logger.exception("Erro ao gerar apostas de refino")
        await update.message.reply_text(f"❌ Erro no /refino: {e}")


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
    app.add_handler(CommandHandler("bolao", bolao_cmd))
    app.add_handler(CommandHandler("refino", refino_cmd))  
    app.add_handler(CommandHandler("confirmar", confirmar_cmd))
    app.add_handler(CommandHandler("avaliar", avaliar_cmd))
    app.add_handler(CommandHandler("status_penalidades", status_penalidades_cmd))

    logger.info("Bot Lotomania (Sistema Inteligente com Penalidades) iniciado.")
    logger.info("NOVO comando /refino disponível - Versão ULTRA-EFICIENTE")
    logger.info("MELHORIAS APLICADAS: Penalidades reforçadas e ausência total de repetição")
    app.run_polling()


if __name__ == "__main__":
    main()
