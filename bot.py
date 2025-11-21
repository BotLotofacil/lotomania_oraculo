import json
import time
import os
import csv
import logging
from typing import List, Set

import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Caminho do histórico e do modelo
HISTORY_PATH = "lotomania_historico_onehot.csv"
MODEL_PATH = "lotomania_model.npz"

ULTIMA_GERACAO_PATH = "lotomania_ultima_geracao.json"
DESEMPENHO_PATH = "lotomania_desempenho.csv"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Arquivo para guardar o último lote gerado (para o /confirmar)
ULTIMA_GERACAO_PATH = "ultima_geracao_oraculo.json"

# Arquivo de telemetria de desempenho
DESEMPENHO_PATH = "desempenho_oraculo.csv"


# ---------------------------
# REDE NEURAL SIMPLES (MLP)
# ---------------------------

class SimpleMLP:
    """
    MLP simples: [input] -> [hidden tanh] -> [sigmoid]
    para prever P(dezena sair no próximo concurso).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, lr: float = 0.01, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = rng.normal(0, 0.1, size=(hidden_dim, 1))
        self.b2 = np.zeros((1,))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = self._tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self._sigmoid(z2)
        return y_hat, a1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80, batch_size: int = 512):
        """
        Treino com mini-batches. Se o modelo já tiver pesos treinados,
        esse método CONTINUA o aprendizado (não reseta).
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]

        for _ in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuff = X[indices]
            y_shuff = y[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                Xb = X_shuff[start:end]
                yb = y_shuff[start:end]
                if Xb.shape[0] == 0:
                    continue

                # forward
                y_hat, a1 = self._forward(Xb)

                # gradiente (binary cross-entropy)
                m = Xb.shape[0]
                dz2 = (y_hat - yb) / m
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (1 - a1 ** 2)

                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)

                # atualização
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_hat, _ = self._forward(X)
        return y_hat.ravel()


class ModelWrapper:
    def __init__(self, mlp: SimpleMLP, mean_: np.ndarray, std_: np.ndarray):
        self.mlp = mlp
        self.mean_ = mean_
        self.std_ = std_


_model_cache: ModelWrapper | None = None  # cache na memória


# ---------------------------
# LEITURA DO HISTÓRICO
# ---------------------------

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


# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

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


def build_dataset(history: List[Set[int]]):
    """
    Gera X, y para treino:
    - X: features da dezena em cada concurso (menos o último)
    - y: 1 se a dezena saiu no concurso seguinte, 0 se não
    """
    X_list = []
    y_list = []

    n = len(history)
    if n < 2:
        raise ValueError("Histórico insuficiente para treinamento.")

    for idx in range(n - 1):
        proximo = history[idx + 1]
        for dezena in range(100):  # 00 a 99
            feats = compute_features_for_dozen(history, idx, dezena)
            X_list.append(feats)
            y_list.append(1 if dezena in proximo else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def compute_scaler(X: np.ndarray):
    mean_ = X.mean(axis=0)
    std_ = X.std(axis=0)
    std_[std_ == 0] = 1.0
    return mean_, std_


# ---------------------------
# SALVAR / CARREGAR MODELO
# ---------------------------

def save_model(wrapper: ModelWrapper, path: str = MODEL_PATH):
    np.savez(
        path,
        W1=wrapper.mlp.W1,
        b1=wrapper.mlp.b1,
        W2=wrapper.mlp.W2,
        b2=wrapper.mlp.b2,
        mean=wrapper.mean_,
        std=wrapper.std_,
    )


def load_model_local(path: str = MODEL_PATH) -> ModelWrapper:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(path):
        raise FileNotFoundError("Modelo ainda não treinado. Use /treinar.")

    data = np.load(path)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    mean_ = data["mean"]
    std_ = data["std"]

    mlp = SimpleMLP(input_dim=W1.shape[0], hidden_dim=W1.shape[1])
    mlp.W1 = W1
    mlp.b1 = b1
    mlp.W2 = W2
    mlp.b2 = b2

    wrapper = ModelWrapper(mlp, mean_, std_)
    _model_cache = wrapper
    return wrapper


# ---------------------------
# GERAÇÃO DE APOSTAS
# ---------------------------

def gerar_probabilidades_para_proximo(history: List[Set[int]], model: ModelWrapper) -> np.ndarray:
    if len(history) < 2:
        raise ValueError("Histórico insuficiente para prever o próximo concurso.")

    idx = len(history) - 1
    feats_list = []

    for dezena in range(100):
        feats = compute_features_for_dozen(history, idx, dezena)
        feats_list.append(feats)

    X = np.array(feats_list, dtype=np.float32)
    X_scaled = (X - model.mean_) / model.std_
    probs = model.mlp.predict_proba(X_scaled)
    return probs  # shape (100,)


def gerar_apostas_e_espelhos(history: List[Set[int]], model: ModelWrapper):
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

def treino_incremental_pos_concurso(history: List[Set[int]], resultado_set: set[int]):
    """
    Treino incremental leve:
    - usa o ÚLTIMO ponto da linha do tempo do histórico
    - calcula features de cada dezena (00–99)
    - faz um passo de treino com o resultado confirmado
    """
    try:
        wrapper = load_model_local()
    except FileNotFoundError:
        # ainda não existe modelo → nada a fazer
        logger.warning("Treino incremental ignorado: modelo ainda não treinado (/treinar).")
        return
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo para treino incremental: {e}")
        return

    if len(history) < 2:
        logger.warning("Histórico insuficiente para treino incremental.")
        return

    idx = len(history) - 1  # último concurso do histórico
    X_list = []
    y_list = []

    for dezena in range(100):
        feats = compute_features_for_dozen(history, idx, dezena)
        X_list.append(feats)
        y_list.append(1 if dezena in resultado_set else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # usa o mesmo scaler original
    X_scaled = (X - wrapper.mean_) / wrapper.std_

    # passo rápido de ajuste
    wrapper.mlp.fit(X_scaled, y, epochs=15, batch_size=100)

    # salva e atualiza cache
    save_model(wrapper)
    global _model_cache
    _model_cache = wrapper

    logger.info("Treino incremental pós-concurso concluído.")


def gerar_apostas_oraculo_supremo(history: List[Set[int]], model: ModelWrapper):
    """
    Oráculo Supremo:
      1 – Repetição
      2 – Ciclos
      3 – Probabilística
      4 – Híbrida
      5 – Dezenas quentes
      6 – Dezenas frias
    Retorna (apostas, espelhos).
    """
    if len(history) < 5:
        raise ValueError("Histórico insuficiente para Oráculo Supremo (mínimo 5 concursos).")

    # Probabilidades da rede para o PRÓXIMO concurso
    probs = gerar_probabilidades_para_proximo(history, model)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()

    n_hist = len(history)
    ultimo_concurso = history[-1]  # conjunto de dezenas do último sorteio

    # --------- Frequência (quentes/frias) em janela recente ---------
    janela = min(200, n_hist)
    recentes = history[-janela:]
    freq = np.zeros(100, dtype=np.int32)
    for conc in recentes:
        for d in conc:
            freq[d] += 1

    # --------- Atrasos (ciclos) ---------
    atrasos = np.zeros(100, dtype=np.int32)
    for dez in range(100):
        gap = 0
        for conc in reversed(history):
            if dez in conc:
                break
            gap += 1
        atrasos[dez] = gap

    rng = np.random.default_rng()

    # Pequeno ruído para evitar apostas sempre idênticas
    probs_ruido = probs + rng.normal(0, 0.01, size=probs.shape)
    probs_ruido = np.clip(probs_ruido, 1e-9, None)
    probs_ruido = probs_ruido / probs_ruido.sum()

    # --------- APOSTA 1 – REPETIÇÃO ---------
    candidatos_rep = sorted(list(ultimo_concurso), key=lambda d: probs[d], reverse=True)
    aposta1 = candidatos_rep[:30]  # até 30 repetidas

    restantes1 = [int(d) for d in np.argsort(-probs) if d not in aposta1]
    aposta1 += restantes1[:(50 - len(aposta1))]
    aposta1 = sorted(aposta1)

    # --------- APOSTA 2 – CICLOS ---------
    score_ciclo = probs * (1.0 + atrasos / (n_hist + 1.0))
    idx_ciclos = np.argsort(score_ciclo)[-50:]
    aposta2 = sorted(idx_ciclos.tolist())

    # --------- APOSTA 3 – PROBABILÍSTICA PURA ---------
    aposta3 = sorted(rng.choice(100, size=50, replace=False, p=probs_ruido).tolist())

    # --------- APOSTA 4 – HÍBRIDA (TOP + MÉDIAS) ---------
    ordem_pred = np.argsort(probs)
    top = ordem_pred[-30:]          # mais prováveis
    medios = ordem_pred[20:80]      # faixa intermediária

    top_list = list(top)
    pool_medios = [int(d) for d in medios if d not in top_list]
    qtd_top = 25
    qtd_medios = 50 - qtd_top

    if len(pool_medios) >= qtd_medios:
        extra4 = rng.choice(pool_medios, size=qtd_medios, replace=False).tolist()
    else:
        extra4 = pool_medios

    aposta4 = sorted(top_list[-qtd_top:] + extra4)

    # --------- APOSTA 5 – DEZENAS QUENTES ---------
    idx_quentes = np.argsort(freq)[-50:]
    aposta5 = sorted(idx_quentes.tolist())

    # --------- APOSTA 6 – DEZENAS FRIAS ---------
    idx_frias = np.argsort(freq)[:50]
    aposta6 = sorted(idx_frias.tolist())

    apostas = [aposta1, aposta2, aposta3, aposta4, aposta5, aposta6]

    # --------- ESPELHOS: complemento 00–99 ---------
    universo = set(range(100))
    espelhos = [sorted(list(universo.difference(set(ap)))) for ap in apostas]

    return apostas, espelhos


def format_dezenas_sortidas(dezenas):
    return " ".join(f"{d:02d}" for d in sorted(dezenas))


# ---------------------------
# HANDLERS TELEGRAM
# ---------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🔮 Oráculo Lotomania\n\n"
        "/treinar - treina ou atualiza a rede neural\n"
        "/gerar - Oráculo Supremo (6 apostas + 6 espelhos)\n"
        "/errar_tudo - gera 3 apostas tentando errar tudo\n"
        "/confirmar - confronta o resultado oficial com o último bloco gerado\n\n"
        "Mantenha o arquivo lotomania_historico_onehot.csv sempre atualizado."
    )
    await update.message.reply_text(msg)
    

async def confirmar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /confirmar 02 08 15 20 24 25 30 34 37 40 43 51 60 62 67 77 81 85 87 94

    1) Lê a última geração salva pelo /gerar
    2) Compara o resultado com as 6 apostas + 6 espelhos
    3) Salva histórico de acertos em CSV
    4) Dispara treino incremental da rede neural
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
        await update.message.reply_text("❌ Não consegui interpretar as dezenas. Use apenas números separados por espaço.")
        return

    # ----------------------------------
    # 2) Carrega última geração
    # ----------------------------------
    if not os.path.exists(ULTIMA_GERACAO_PATH):
        await update.message.reply_text(
            "⚠️ Arquivo de última geração não encontrado.\n"
            "Gere um novo bloco com /gerar e depois use /confirmar."
        )
        return

    try:
        with open(ULTIMA_GERACAO_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)

        apostas = dados.get("apostas")
        espelhos = dados.get("espelhos")

        if not apostas or not espelhos:
            raise ValueError("Dados incompletos na última geração.")

        # garante int nativo
        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

    except Exception as e:
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

    melhor_ap_idx = int(np.argmax(hits_apostas))  # 0–5
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
                ]
                writer.writerow(header)

            ts = time.time()
            resultado_txt = " ".join(f"{d:02d}" for d in sorted(resultado))

            row = [
                f"{ts:.3f}",
                resultado_txt,
                *[int(h) for h in hits_apostas],
                *[int(h) for h in hits_espelhos],
                melhor_ap_idx + 1,
                melhor_esp_idx + 1,
            ]
            writer.writerow(row)

        logger.info("Desempenho registrado em %s", DESEMPENHO_PATH)

    except Exception as e_csv:
        logger.exception("Erro ao salvar desempenho em CSV: %s", e_csv)

    # ----------------------------------
    # 5) Treino incremental da rede neural
    # ----------------------------------
    try:
        history = load_history(HISTORY_PATH)
        treino_incremental_pos_concurso(history, resultado_set)
        txt_treino = "\n🧠 Treino incremental aplicado ao modelo."
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

    labels = [
        "Aposta 1 – Repetição",
        "Aposta 2 – Ciclos",
        "Aposta 3 – Probabilística",
        "Aposta 4 – Híbrida",
        "Aposta 5 – Dezenas quentes",
        "Aposta 6 – Dezenas frias",
    ]

    for i in range(6):
        linhas.append(f"{labels[i]}: {hits_apostas[i]} acertos")
        linhas.append(f"Espelho {i+1}: {hits_espelhos[i]} acertos")
        linhas.append("")

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
        await update.message.reply_text("🧠 Iniciando treinamento com o histórico...")

        history = load_history(HISTORY_PATH)
        X, y = build_dataset(history)
        mean_, std_ = compute_scaler(X)
        X_scaled = (X - mean_) / std_

        # Se já existe modelo, continua o treino (não reseta)
        if os.path.exists(MODEL_PATH):
            wrapper_antigo = load_model_local()
            mlp = wrapper_antigo.mlp
        else:
            mlp = SimpleMLP(input_dim=X.shape[1], hidden_dim=32, lr=0.01)

        mlp.fit(X_scaled, y, epochs=60, batch_size=512)

        wrapper = ModelWrapper(mlp, mean_, std_)
        save_model(wrapper)

        await update.message.reply_text(f"✅ Treinamento concluído. Amostras usadas: {len(y)}")

    except Exception as e:
        logger.exception("Erro no treinamento")
        await update.message.reply_text(f"❌ Erro no treinamento: {e}")


async def gerar_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        history = load_history(HISTORY_PATH)
        model = load_model_local()

        # Usa o Oráculo Supremo
        apostas, espelhos = gerar_apostas_oraculo_supremo(history, model)

        # CONVERSÃO: transforma tudo em int nativo do Python
        apostas_py = [[int(x) for x in ap] for ap in apostas]
        espelhos_py = [[int(x) for x in esp] for esp in espelhos]

        # Salva última geração para o /confirmar
        try:
            dados = {
                "timestamp": float(time.time()),
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

        linhas = ["🙃 *Apostas para tentar errar tudo*\n"]
        for i, ap in enumerate(apostas_erro, start=1):
            linhas.append(f"Aposta erro {i}: {format_dezenas_sortidas(ap)}")

        await update.message.reply_markdown("\n".join(linhas))

    except Exception as e:
        logger.exception("Erro ao gerar apostas de erro")
        await update.message.reply_text(f"❌ Erro ao gerar apostas de erro: {e}")


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

    logger.info("Bot Lotomania iniciado.")
    app.run_polling()


if __name__ == "__main__":
    main()
