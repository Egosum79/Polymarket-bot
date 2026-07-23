#!/usr/bin/env python3
"""
=======================================================
  RECALIBRADOR DE MODELO
  Ajusta los pesos de la señal de btc_direction_bot.py y
  btc_scalp_bot.py a partir de resultados YA liquidados
  (settlements.jsonl), en vez de los pesos que se pusieron
  a mano al construir los bots.
=======================================================

QUÉ ES (y qué NO es):
  Esto es aprendizaje estadístico clásico — una regresión
  logística simple, ajustada por descenso de gradiente, sobre
  los mismos indicadores que ya calculan los bots (RSI, EMA,
  momentum de ventana). NO es un agente de IA ni un LLM: es la
  herramienta correcta y auditable para calibrar pesos numéricos
  a partir de datos estructurados, no una caja negra.

CÓMO FUNCIONA:
  1. Lee btc_bot_log.jsonl / btc_scalp_log.jsonl + settlements.jsonl
  2. Para cada apuesta ya liquidada, reconstruye cuál fue la
     dirección REAL de BTC (no si ganamos — si subió o bajó),
     y los valores de los indicadores en el momento de la señal
  3. Si hay al menos MIN_SAMPLES apuestas liquidadas para ese bot,
     ajusta P(UP) = sigmoid(w0 + w1*x1 + ... ) por descenso de
     gradiente sobre esos datos
  4. Guarda los pesos en bot2_weights.json / bot3_weights.json
  5. Si NO hay suficientes muestras todavía, no toca nada — los
     bots siguen usando su heurística original hasta entonces

  btc_direction_bot.py y btc_scalp_bot.py leen estos archivos de
  pesos si existen y los usan en vez de la heurística fija — sin
  necesidad de ningún cambio manual cuando eso pase. Pensado para
  correr automáticamente cada día (ver daily-review.yml); mientras
  no haya datos suficientes, simplemente no hace nada.

⚠️  MUESTRA CHICA = DESCONFIAR:
  Con menos de ~100 muestras, un modelo ajustado puede estar
  memorizando ruido, no una señal real. Esto se refleja en el
  log de salida (avisa si la muestra sigue siendo chica) pero NO
  bloquea el ajuste — es información para interpretar el reporte
  diario, no una garantía de calidad del modelo.

USO:
  python retrain_model.py
"""

import json
import math
import sys
from datetime import datetime, timezone

from settle_bets import load_jsonl

# En consolas Windows con codepage legado (cp1252), imprimir emojis revienta
# con UnicodeEncodeError. Forzamos stdout/stderr a UTF-8 si el terminal lo permite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SETTLEMENTS_FILE = "settlements.jsonl"
MIN_SAMPLES      = 40    # piso mínimo para intentar un ajuste (ver aviso arriba)
CONFIDENT_AT      = 100  # a partir de aquí se avisa que la muestra ya es más sólida
EPOCHS           = 3000
LEARNING_RATE    = 0.3


# ─────────────────────────────────────────────────────
# REGRESIÓN LOGÍSTICA (descenso de gradiente, sin dependencias)
# ─────────────────────────────────────────────────────

def sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))   # evita overflow en math.exp
    return 1.0 / (1.0 + math.exp(-z))


def standardize(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    n_features = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(n_features)]
    stds  = []
    for j in range(n_features):
        var = sum((row[j] - means[j]) ** 2 for row in X) / len(X)
        stds.append(math.sqrt(var) or 1.0)   # evita división por 0 si una feature es constante
    X_std = [[(row[j] - means[j]) / stds[j] for j in range(n_features)] for row in X]
    return X_std, means, stds


def fit_logistic(X: list[list[float]], y: list[int],
                  epochs: int = EPOCHS, lr: float = LEARNING_RATE) -> list[float]:
    """Ajusta w0 + w1*x1 + ... por descenso de gradiente batch. X ya estandarizado."""
    n_features = len(X[0])
    n = len(X)
    weights = [0.0] * (n_features + 1)   # weights[0] = intercepto

    for _ in range(epochs):
        grad = [0.0] * (n_features + 1)
        for xi, yi in zip(X, y):
            z = weights[0] + sum(w * x for w, x in zip(weights[1:], xi))
            error = sigmoid(z) - yi
            grad[0] += error
            for j in range(n_features):
                grad[j + 1] += error * xi[j]
        for j in range(n_features + 1):
            weights[j] -= lr * grad[j] / n

    return weights


# ─────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATASET DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────

def build_dataset(log_entries: list[dict], settlements: list[dict], bot_name: str,
                   feature_fn) -> tuple[list[list[float]], list[int]]:
    """
    Une cada señal BET del log con su resultado ya liquidado (por market_id +
    timestamp) y aplica feature_fn para extraer las variables de entrada.
    """
    settled_index = {
        (s.get("market_id"), s.get("timestamp")): s
        for s in settlements if s.get("bot") == bot_name
    }
    X, y = [], []
    for e in log_entries:
        if e.get("action") != "BET":
            continue
        key = (e.get("market_id"), e.get("timestamp"))
        s = settled_index.get(key)
        if s is None:
            continue   # todavía no liquidada
        features = feature_fn(e)
        if features is None:
            continue   # faltan datos para esta entrada (ej. sin window_momentum)
        actual_up = (e.get("bet_side") == "UP") == bool(s.get("won"))
        X.append(features)
        y.append(1 if actual_up else 0)
    return X, y


def features_bot2(e: dict) -> list[float] | None:
    rsi = e.get("rsi")
    if rsi is None:
        return None
    ema_signal  = 1.0 if e.get("ema_signal") == "UP" else -1.0
    macd_signal = 1.0 if e.get("macd_signal") == "UP" else -1.0
    momentum    = e.get("momentum_1h", 0) or 0
    return [rsi, ema_signal, macd_signal, momentum]


def features_bot3(e: dict) -> list[float] | None:
    rsi = e.get("rsi")
    ema_diff = e.get("ema_diff")
    wm = e.get("window_momentum")
    if rsi is None or ema_diff is None or wm is None:
        return None
    return [rsi, ema_diff, wm]


# ─────────────────────────────────────────────────────
# ENTRENAR Y GUARDAR UN BOT
# ─────────────────────────────────────────────────────

def retrain_bot(log_path: str, bot_name: str, feature_names: list[str],
                 feature_fn, weights_path: str, settlements: list[dict]):
    print(f"\n── {bot_name} ({log_path}) ──")
    log_entries = load_jsonl(log_path)
    X, y = build_dataset(log_entries, settlements, bot_name, feature_fn)

    print(f"  Muestras liquidadas disponibles: {len(X)}")
    if len(X) < MIN_SAMPLES:
        print(f"  ⚪ Todavía no hay suficientes ({MIN_SAMPLES} mínimo) — "
              f"sigue con la heurística original.")
        return

    if len(X) < CONFIDENT_AT:
        print(f"  ⚠️  Muestra todavía chica (< {CONFIDENT_AT}) — se ajusta igual, "
              f"pero interpreta el resultado con cautela.")

    X_std, means, stds = standardize(X)
    weights = fit_logistic(X_std, y)

    # Precisión sobre los mismos datos de entrenamiento (informativo, no es
    # validación fuera de muestra — solo para ver que el ajuste no delira).
    correct = 0
    for xi, yi in zip(X_std, y):
        z = weights[0] + sum(w * x for w, x in zip(weights[1:], xi))
        pred = 1 if sigmoid(z) >= 0.5 else 0
        correct += (pred == yi)
    train_accuracy = correct / len(y)

    output = {
        "bot":            bot_name,
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "n_samples":      len(X),
        "features":       feature_names,
        "means":          means,
        "stds":           stds,
        "weights":        weights,   # [intercepto, w1, w2, ...] sobre features estandarizadas
        "train_accuracy": round(train_accuracy, 4),
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Modelo reajustado con {len(X)} muestras → {weights_path}")
    print(f"     Precisión sobre datos de entrenamiento: {train_accuracy*100:.1f}%")


def main():
    print("=" * 60)
    print("  RECALIBRADOR DE MODELO (aprendizaje estadístico)")
    print("=" * 60)

    settlements = load_jsonl(SETTLEMENTS_FILE)
    print(f"  Total de apuestas liquidadas en el sistema: {len(settlements)}")

    retrain_bot(
        log_path="btc_bot_log.jsonl", bot_name="bot2",
        feature_names=["rsi", "ema_signal", "macd_signal", "momentum_1h"],
        feature_fn=features_bot2, weights_path="bot2_weights.json",
        settlements=settlements,
    )
    retrain_bot(
        log_path="btc_scalp_log.jsonl", bot_name="bot3",
        feature_names=["rsi", "ema_diff", "window_momentum"],
        feature_fn=features_bot3, weights_path="bot3_weights.json",
        settlements=settlements,
    )

    print("\n  ✅ Recalibración completada.")


if __name__ == "__main__":
    main()
