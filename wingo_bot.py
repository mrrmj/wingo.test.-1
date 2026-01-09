from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import requests, time, numpy as np
from sklearn.ensemble import RandomForestClassifier

BOT_TOKEN = "8397157307:AAEswhlHJelvZ0kHv0cRlIWvJ7awbdRw3IM"
history = []
last_prediction = None

# ====================== ML and Wingo Functions ======================

def preload_history():
    url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    params = {"ts": int(time.time() * 1000)}
    data = requests.get(url, params=params).json()["data"]["list"]
    history.clear()
    for item in data[:50]:  # ✅ Correct order: oldest to newest
        history.append((int(item["number"]), item["color"]))

    # Debug: Check balance
    print("=== History Stats ===")
    small = big = red = green = violet = 0
    for n, c in history:
        if n <= 4: small += 1
        else: big += 1
        cl = c.lower()
        if "red" in cl: red += 1
        if "green" in cl: green += 1
        if "violet" in cl: violet += 1
    print(f"Small: {small}, Big: {big}")
    print(f"Red: {red}, Green: {green}, Violet: {violet}")

def get_running_number():
    url = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    data = requests.get(url, params={"ts": int(time.time() * 1000)}).json()["data"]["list"][0]
    period = data["issueNumber"]
    number = int(data["number"])
    color = data["color"]
    if not history or history[-1][0] != number or history[-1][1] != color:
        history.append((number, color))
    if len(history) > 100:
        history.pop(0)
    return period, number, color

# ✅ Updated Size/Color Logic as per your rule
def get_size(n): return "Big" if n >= 5 else "Small"
def get_color(n): return "Violet" if n in [0, 13, 26] else ("Red" if n % 2 == 0 else "Green")
def get_size_code(n): return 1 if n >= 5 else 0
def get_color_code(n): return 2 if n in [0, 13, 26] else (0 if n % 2 == 0 else 1)

def train_classifiers():
    if len(history) < 10:
        return None, None

    X, y_size, y_color = [], [], []
    for i in range(len(history) - 5):
        window = [n for n, _ in history[i:i + 5]]
        next_number = history[i + 5][0]
        X.append(window)
        y_size.append(get_size_code(next_number))
        y_color.append(get_color_code(next_number))

    size_data = list(zip(X, y_size, y_color))
    big_data = [d for d in size_data if d[1] == 1]
    small_data = [d for d in size_data if d[1] == 0]
    min_len = min(len(big_data), len(small_data))

    if min_len >= 3:
        # ✅ Use balanced data
        balanced = big_data[:min_len] + small_data[:min_len]
        X_bal = [x for x, _, _ in balanced]
        y_bal_size = [s for _, s, _ in balanced]
        y_bal_color = [c for _, _, c in balanced]
        print(f"✅ Balanced training → Big: {min_len}, Small: {min_len}")
    else:
        # ⚠️ Fallback: Use full unbalanced data
        X_bal = X
        y_bal_size = y_size
        y_bal_color = y_color
        print(f"⚠️ Unbalanced training → Big: {len(big_data)}, Small: {len(small_data)}")

    size_model = RandomForestClassifier(n_estimators=100)
    color_model = RandomForestClassifier(n_estimators=100)
    size_model.fit(np.array(X_bal), y_bal_size)
    color_model.fit(np.array(X_bal), y_bal_color)

    return size_model, color_model


def predict_next(size_model, color_model):
    if len(history) < 5 or size_model is None: return None
    X_test = np.array([[n for n, _ in history[-5:]]])
    print(f"=== Predicting Next From: {X_test[0]}")
    size_code = size_model.predict(X_test)[0]
    color_code = color_model.predict(X_test)[0]
    size_str = "Big" if size_code == 1 else "Small"
    color_str = ["Red", "Green", "Violet"][color_code]
    print(f"Prediction: {size_str} / {color_str}")
    return size_str, color_str

# ====================== Telegram Bot Handlers ======================

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎉 Welcome! Use /predict to get the next prediction.")

async def predict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global last_prediction

    period, number, color_raw = get_running_number()
    next_period = str(int(period) + 1)

    actual_size = get_size(number).lower()
    actual_color = get_color(number).lower()
    color_raw_lower = color_raw.lower()  # "green,violet" etc.

    size_model, color_model = train_classifiers()
    prediction = predict_next(size_model, color_model)

    result_msg = "No previous prediction to check."
    if last_prediction:
        pred_period, pred_size, pred_color = last_prediction
        if int(pred_period) > int(period):
            result_msg = "⏳ Waiting for result..."
        else:
            pred_size = pred_size.lower()
            pred_color = pred_color.lower()
            size_result = "✅ Win" if pred_size == actual_size else "❌ Loss"
            color_result = "✅ Win" if pred_color in color_raw_lower else "❌ Loss"
            result_msg = (
                f"🧾 Previous Prediction:\n"
                f"Size: {pred_size.capitalize()} → {actual_size.capitalize()} → {size_result}\n"
                f"Color: {pred_color.capitalize()} → {color_raw} → {color_result}"
            )

    if prediction:
        pred_size, pred_color = prediction
        last_prediction = (next_period, pred_size, pred_color)
        await update.message.reply_text(
            f"{result_msg}\n\n"
            f"🕑 Period: {period}\n🎯 Last Number: {number} ({color_raw})\n\n"
            f"📢 Next Period: {next_period}\n🔮 Prediction: {pred_size} / {pred_color}"
        )
    else:
        await update.message.reply_text("Not enough data to predict.")

# ====================== Run Bot ======================

preload_history()
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))
app.run_polling()

