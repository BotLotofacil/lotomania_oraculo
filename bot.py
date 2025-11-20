import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from modelo import treinar_modelo, gerar_aposta, gerar_aposta_espelho, gerar_errar_tudo

TOKEN = "COLOQUE_SEU_TOKEN_AQUI"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Oráculo Lotomania ativado!")

async def treinar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = treinar_modelo()
    await update.message.reply_text("🧠 " + msg)

async def gerar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    a1 = gerar_aposta()
    a2 = gerar_aposta()
    a3 = gerar_aposta()

    e1 = gerar_aposta_espelho()
    e2 = gerar_aposta_espelho()
    e3 = gerar_aposta_espelho()

    texto = (
        "🔮 *Apostas do Oráculo:*\n\n"
        f"Aposta 1: {a1}\n"
        f"Aposta 2: {a2}\n"
        f"Aposta 3: {a3}\n\n"
        "♻ *Espelhos*\n"
        f"Espelho 1: {e1}\n"
        f"Espelho 2: {e2}\n"
        f"Espelho 3: {e3}\n"
    )
    await update.message.reply_text(texto)

async def errar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    aposta = gerar_errar_tudo()
    await update.message.reply_text(f"❌ Aposta para errar tudo:\n{aposta}")

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("treinar", treinar))
    app.add_handler(CommandHandler("gerar", gerar))
    app.add_handler(CommandHandler("errar_tudo", errar))

    app.run_polling()

if __name__ == "__main__":
    main()
