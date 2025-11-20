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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def format_dezenas_sortidas(dezenas):
    return " ".join(f"{d:02d}" for d in sorted(dezenas))


# ---------------------------
# HANDLERS TELEGRAM
# ---------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🔮 *Oráculo Lotomania*\n\n"
        "/treinar - treina ou atualiza a rede neural\n"
        "/gerar - gera 3 apostas + 3 espelhos\n"
        "/errar_tudo - gera 3 apostas tentando errar tudo\n\n"
        "Certifique-se de manter o arquivo lotomania_historico_onehot.csv atualizado."
    )
    await update.message.reply_markdown(msg)


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

        apostas, espelhos = gerar_apostas_e_espelhos(history, model)

        linhas = ["🔮 *Apostas sugeridas (Lotomania)*\n"]
        for i, ap in enumerate(apostas, start=1):
            linhas.append(f"*Aposta {i}:*  {format_dezenas_sortidas(ap)}")
            linhas.append(f"Espelho {i}: {format_dezenas_sortidas(espelhos[i-1])}")
            linhas.append("")

        await update.message.reply_markdown("\n".join(linhas))

    except Exception as e:
        logger.exception("Erro ao gerar apostas")
        await update.message.reply_text(f"❌ Erro ao gerar apostas: {e}")


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

    logger.info("Bot Lotomania iniciado.")
    app.run_polling()


if __name__ == "__main__":
    main()
