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
     recorta cada feature a su rango [percentil 1, percentil 99] visto
     en el entrenamiento (evita que un valor en vivo mucho más extremo
     que todo lo visto antes dispare una probabilidad sobreconfiada —
     auditoría 2026-07-27), separa un HOLDOUT cronológico (las muestras
     más recientes, nunca vistas al elegir el modelo) para probar varias
     intensidades de regularización L2 y quedarse con la que mejor
     generaliza — no la que mejor memoriza el propio entrenamiento — y
     ajusta P(UP) = sigmoid(w0 + w1*x1 + ...) por descenso de gradiente
  4. Guarda los pesos (+ límites de recorte + L2 elegido + precisión
     fuera de muestra) en bot2_weights.json / bot3_weights.json
  5. Si NO hay suficientes muestras todavía, no toca nada — los
     bots siguen usando su heurística original hasta entonces
  6. No reentrena más de una vez cada MIN_RETRAIN_INTERVAL_HOURS,
     sin importar cuántas veces se dispare el workflow ese día (ver
     aviso de cadencia más abajo)

  btc_direction_bot.py y btc_scalp_bot.py leen estos archivos de
  pesos si existen y los usan en vez de la heurística fija — sin
  necesidad de ningún cambio manual cuando eso pase. Pensado para
  correr automáticamente cada día (ver daily-review.yml); mientras
  no haya datos suficientes, simplemente no hace nada.

CALIBRACIÓN DE ESPORTS (auditoría 2026-09-07):
  esports_bot.py no tiene features técnicas que reentrenar como los bots
  de BTC -- su heurística (forma reciente 40% + h2h 30% + tier 20% +
  región 10%) SÍ ordena bien a los equipos (más probabilidad predicha =
  más aciertos reales, de forma monótona en 1420 apuestas), pero está
  sistemáticamente sobreconfiada: cuando dice 70% de probabilidad, en
  la práctica acierta ~38%. En vez de reentrenar los pesos de la mezcla,
  se ajusta un calibrador Platt (regresión logística de UNA sola variable,
  logit(probabilidad cruda) → probabilidad real) reutilizando el mismo
  pipeline de arriba (fit_logistic + elegir_l2 tratan esto como un
  problema de 1 feature). Simulado contra el historial completo con el
  mismo umbral de edge, esto cambia el PnL de -$149 a +$455. Se guarda en
  esports_calibration.json; esports_bot.py lo aplica si existe, y si no,
  sigue usando la probabilidad cruda sin corregir (mismo comportamiento
  de siempre).

BOOTSTRAP HISTÓRICO PARA BOT3 (investigación 2026-09-16):
  Con solo ~700 apuestas liquidadas, bot3_weights.json llegaba apenas a
  58.99% de precisión fuera de muestra -- forzado a ser tímido para no
  sobreajustarse con tan poca muestra (nunca se aleja mucho de 50/50, ver
  auditoría del veto de momentum). Reconstruyendo las mismas 3 variables
  (rsi, ema_diff, window_momentum) para CADA ventana de 15 min de los
  últimos HIST_DIAS_BOT3 días -- calculable directo del histórico de
  precios de Binance, sin necesitar que el bot haya apostado ahí -- se
  consiguen miles de muestras en vez de cientos. El mismo modelo de 3
  variables, entrenado así, sube a 70.41% fuera de muestra. Por eso
  build_historical_windows_bot3() genera este dataset y se concatena con
  las apuestas reales ya liquidadas antes de ajustar bot3_weights.json --
  no reemplaza los datos reales, los complementa. Bot2 y esports no usan
  esto: para Bot2 (mercados horarios) reconstruir el equivalente exigiría
  datos de MACD/volumen/momentum a escala horaria no validados aún, y
  esports no tiene features técnicas que reconstruir de esta forma.

⚠️  MUESTRA CHICA = DESCONFIAR:
  Con menos de ~100 muestras, un modelo ajustado puede estar
  memorizando ruido, no una señal real. Esto se refleja en el
  log de salida (avisa si la muestra sigue siendo chica) pero NO
  bloquea el ajuste — es información para interpretar el reporte
  diario, no una garantía de calidad del modelo.

⚠️  CADENCIA DE REENTRENAMIENTO (auditoría 2026-07-29):
  daily-review.yml permite disparo manual (workflow_dispatch), y cada
  disparo corría este script de nuevo — varios reentrenamientos el
  mismo día agregan ruido (los coeficientes flotan entre corridas con
  apenas 1-2 muestras nuevas). Este script ahora revisa el
  "trained_at" del archivo de pesos existente y se salta el ajuste si
  ya se reentrenó hace menos de MIN_RETRAIN_INTERVAL_HOURS, sin
  importar cuántas veces se ejecute el workflow.

USO:
  python retrain_model.py
"""

import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from settle_bets import load_jsonl
from btc_direction_bot import ema, rsi   # mismas funciones que usa el bot en vivo

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
L2_GRID          = [0.0, 0.1, 0.3, 1.0, 3.0]   # candidatos de regularizacion L2 a probar
HOLDOUT_FRACTION = 0.2                          # fraccion (cronologica, mas reciente) reservada para validar
MIN_RETRAIN_INTERVAL_HOURS = 20                 # no reentrenar mas seguido que esto
ESPORTS_MIN_SAMPLES = 100   # la heuristica de esports es mas ruidosa que las de BTC

# ── Bootstrap historico para bot3 (ver docstring del modulo) ──
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
HIST_DIAS_BOT3 = 60          # dias de velas 1m a reconstruir
HIST_CANDLES_LOOKBACK = 30   # igual a CANDLES_1M en btc_scalp_bot.py
HIST_OFFSET_MIN = 5          # evaluar 5 min despues de abierta cada ventana de 15 min
HIST_DURACION_VENTANA = 15


# ─────────────────────────────────────────────────────
# REGRESIÓN LOGÍSTICA (descenso de gradiente, sin dependencias)
# ─────────────────────────────────────────────────────

def sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))   # evita overflow en math.exp
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float) -> float:
    p = min(0.98, max(0.02, p))
    return math.log(p / (1 - p))


def percentile(values: list[float], p: float) -> float:
    """Percentil p (0-100) por interpolación lineal, sin dependencias externas."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100) * (len(s) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def clip_bounds(X: list[list[float]], low_p: float = 1, high_p: float = 99) -> tuple[list[float], list[float]]:
    """
    Límites [percentil 1, percentil 99] de cada feature, calculados sobre los
    datos de entrenamiento. Ver aviso de sobreconfianza por extrapolación en
    el docstring del módulo: sin esto, un valor en vivo mucho más extremo que
    cualquiera visto durante el ajuste (ej. un momentum de ventana atípico)
    dispara un logit desproporcionado y una probabilidad pegada al techo.
    """
    n_features = len(X[0])
    lows, highs = [], []
    for j in range(n_features):
        col = [row[j] for row in X]
        lows.append(percentile(col, low_p))
        highs.append(percentile(col, high_p))
    return lows, highs


def clip_row(row: list[float], lows: list[float], highs: list[float]) -> list[float]:
    return [max(lows[j], min(highs[j], row[j])) for j in range(len(row))]


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
                  epochs: int = EPOCHS, lr: float = LEARNING_RATE,
                  l2: float = 0.0) -> list[float]:
    """
    Ajusta w0 + w1*x1 + ... por descenso de gradiente batch. X ya estandarizado.
    l2 penaliza la magnitud de w1..wn (no el intercepto) para que el ajuste no
    persiga cada muestra nueva con coeficientes cada vez mas extremos --
    la auditoria 2026-07-29 encontro que los pesos de bot2 cambiaban de signo
    entre reentrenamientos con apenas 2 muestras nuevas.
    """
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
        weights[0] -= lr * grad[0] / n
        for j in range(n_features):
            weights[j + 1] -= lr * (grad[j + 1] / n + l2 * weights[j + 1])

    return weights


def accuracy(weights: list[float], X: list[list[float]], y: list[int]) -> float:
    """Exactitud de clasificacion (umbral 0.5) sobre X/y ya estandarizados."""
    if not y:
        return 0.0
    correct = 0
    for xi, yi in zip(X, y):
        z = weights[0] + sum(w * x for w, x in zip(weights[1:], xi))
        pred = 1 if sigmoid(z) >= 0.5 else 0
        correct += (pred == yi)
    return correct / len(y)


# ─────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATASET DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────

def build_dataset(log_entries: list[dict], settlements: list[dict], bot_name: str,
                   feature_fn) -> tuple[list[list[float]], list[int], list[str]]:
    """
    Une cada señal BET del log con su resultado ya liquidado (por market_id +
    timestamp) y aplica feature_fn para extraer las variables de entrada.
    Tambien devuelve el timestamp de cada señal, para poder separar un
    holdout cronologico (las mas recientes) al elegir la regularizacion.
    """
    settled_index = {
        (s.get("market_id"), s.get("timestamp")): s
        for s in settlements if s.get("bot") == bot_name
    }
    X, y, ts = [], [], []
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
        ts.append(e.get("timestamp") or "")
    return X, y, ts


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
# BOOTSTRAP HISTÓRICO PARA BOT3 (ver docstring del módulo)
# ─────────────────────────────────────────────────────

def _fetch_klines_1m(dias: int) -> list[list]:
    """Descarga velas de 1 minuto de BTCUSDT paginando de a 1000."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - dias * 86_400_000
    paso_ms = 1000 * 60_000  # 1000 velas de 1 min

    todas = []
    cursor = start_ms
    while cursor < end_ms:
        tramo_fin = min(cursor + paso_ms, end_ms)
        url = f"{BINANCE_KLINES_URL}?symbol=BTCUSDT&interval=1m&startTime={cursor}&endTime={tramo_fin}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            cursor = tramo_fin
            continue
        if not data:
            cursor = tramo_fin
            continue
        todas.extend(data)
        cursor = data[-1][0] + 60_000
        time.sleep(0.12)
    return todas


def build_historical_windows_bot3(dias: int = HIST_DIAS_BOT3) -> tuple[list[list[float]], list[int], list[str]]:
    """
    Reconstruye, para cada ventana de 15 min (alineada a :00/:15/:30/:45 UTC)
    de los ultimos `dias` dias, las mismas 3 variables que usa bot3 en vivo
    (rsi, ema_diff, window_momentum) evaluadas HIST_OFFSET_MIN minutos
    despues de abierta la ventana, junto con si el precio subio o bajo al
    cierre. No depende de que el bot haya apostado -- se calcula directo
    del historico real de precios (ver docstring del modulo).
    """
    klines = _fetch_klines_1m(dias)
    if len(klines) < HIST_CANDLES_LOOKBACK + HIST_DURACION_VENTANA + 10:
        return [], [], []

    closes = [float(k[4]) for k in klines]
    times = [k[0] for k in klines]
    n = len(klines)

    X, y, ts = [], [], []
    for i in range(HIST_CANDLES_LOOKBACK, n):
        minuto_del_dia = (times[i] // 60_000) % (24 * 60)
        if minuto_del_dia % HIST_DURACION_VENTANA != HIST_OFFSET_MIN:
            continue

        idx_apertura = i - HIST_OFFSET_MIN
        idx_cierre = i - HIST_OFFSET_MIN + HIST_DURACION_VENTANA
        if idx_cierre >= n:
            continue

        window_open_price = closes[idx_apertura]
        window_close_price = closes[idx_cierre]

        ventana_cerrada = closes[max(0, i - HIST_CANDLES_LOOKBACK + 1):i + 1]
        if len(ventana_cerrada) < HIST_CANDLES_LOOKBACK:
            continue

        rsi_val = rsi(ventana_cerrada)
        ema5 = ema(ventana_cerrada, 5)[-1]
        ema15 = ema(ventana_cerrada, 15)[-1]
        ema_diff = ema5 - ema15
        window_momentum = (ventana_cerrada[-1] - window_open_price) / window_open_price * 100

        X.append([rsi_val, ema_diff, window_momentum])
        y.append(1 if window_close_price > window_open_price else 0)
        ts.append(str(times[i]))

    return X, y, ts


# ─────────────────────────────────────────────────────
# SELECCIÓN DE REGULARIZACIÓN POR HOLDOUT CRONOLÓGICO
# ─────────────────────────────────────────────────────

def elegir_l2(X: list[list[float]], y: list[int], ts: list[str],
              l2_grid: list[float] = L2_GRID,
              holdout_frac: float = HOLDOUT_FRACTION) -> tuple[float, float | None]:
    """
    Ordena las muestras por tiempo y reserva las MAS RECIENTES como holdout
    (nunca vistas al entrenar) -- simula la situación real de predecir hacia
    adelante, no interpolar dentro del mismo set. Prueba cada candidato de
    L2 en ese split y devuelve el que mejor generaliza al holdout, no el que
    mejor memoriza el propio entrenamiento (eso es lo que el "train_accuracy"
    de antes no podía distinguir -- la auditoría lo marcó como
    sistemáticamente optimista).

    Devuelve (l2_elegido, accuracy_holdout). Si la muestra es demasiado
    chica para separar un holdout confiable, devuelve el L2 más conservador
    del grid (mayor regularización) y None en vez de accuracy_holdout.
    """
    orden = sorted(range(len(X)), key=lambda i: ts[i])
    X_ord = [X[i] for i in orden]
    y_ord = [y[i] for i in orden]

    n_holdout = max(1, round(len(X_ord) * holdout_frac))
    n_train = len(X_ord) - n_holdout
    if n_train < 10 or n_holdout < 5:
        return max(l2_grid), None

    X_train, y_train = X_ord[:n_train], y_ord[:n_train]
    X_hold, y_hold = X_ord[n_train:], y_ord[n_train:]

    clip_low, clip_high = clip_bounds(X_train)
    X_train_c = [clip_row(r, clip_low, clip_high) for r in X_train]
    X_hold_c = [clip_row(r, clip_low, clip_high) for r in X_hold]
    X_train_s, means, stds = standardize(X_train_c)
    X_hold_s = [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in X_hold_c]

    mejor_l2, mejor_acc = l2_grid[0], -1.0
    for l2 in l2_grid:
        w = fit_logistic(X_train_s, y_train, l2=l2)
        acc = accuracy(w, X_hold_s, y_hold)
        if acc > mejor_acc:
            mejor_acc, mejor_l2 = acc, l2
    return mejor_l2, mejor_acc


def ya_reentrenado_recientemente(weights_path: str) -> bool:
    """
    Evita reentrenar mas de una vez cada MIN_RETRAIN_INTERVAL_HOURS, sin
    importar cuantas veces se dispare el workflow (ver aviso de cadencia
    en el docstring del modulo).
    """
    try:
        with open(weights_path, encoding="utf-8") as f:
            data = json.load(f)
        trained_at = data.get("trained_at")
        if not trained_at:
            return False
        dt = datetime.fromisoformat(trained_at)
        edad_horas = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        return edad_horas < MIN_RETRAIN_INTERVAL_HOURS
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
        return False


# ─────────────────────────────────────────────────────
# ENTRENAR Y GUARDAR UN BOT
# ─────────────────────────────────────────────────────

def retrain_bot(log_path: str, bot_name: str, feature_names: list[str],
                 feature_fn, weights_path: str, settlements: list[dict],
                 historical_fn=None):
    print(f"\n── {bot_name} ({log_path}) ──")

    if ya_reentrenado_recientemente(weights_path):
        print(f"  ⏳ Ya se reentrenó hace menos de {MIN_RETRAIN_INTERVAL_HOURS}h — "
              f"se omite este ciclo para no acumular ruido.")
        return

    log_entries = load_jsonl(log_path)
    X, y, ts = build_dataset(log_entries, settlements, bot_name, feature_fn)
    print(f"  Apuestas liquidadas disponibles: {len(X)}")

    if historical_fn is not None:
        try:
            X_hist, y_hist, ts_hist = historical_fn()
        except Exception as e:
            X_hist, y_hist, ts_hist = [], [], []
            print(f"  ⚠️  No se pudo reconstruir el histórico de precios: {e}", file=sys.stderr)
        if X_hist:
            print(f"  + {len(X_hist)} ventanas reconstruidas del histórico real de precios "
                  f"(no dependen de que el bot haya apostado — ver docstring del módulo)")
            X = X + X_hist
            y = y + y_hist
            ts = ts + ts_hist

    print(f"  Muestras totales para el ajuste: {len(X)}")
    if len(X) < MIN_SAMPLES:
        print(f"  ⚪ Todavía no hay suficientes ({MIN_SAMPLES} mínimo) — "
              f"sigue con la heurística original.")
        return

    if len(X) < CONFIDENT_AT:
        print(f"  ⚠️  Muestra todavía chica (< {CONFIDENT_AT}) — se ajusta igual, "
              f"pero interpreta el resultado con cautela.")

    l2, oos_accuracy = elegir_l2(X, y, ts)
    if oos_accuracy is not None:
        print(f"  🔎 L2 elegido por validación fuera de muestra: {l2} "
              f"(precisión holdout: {oos_accuracy*100:.1f}%)")
    else:
        print(f"  ⚠️  Muestra insuficiente para separar un holdout confiable — "
              f"usando L2 conservador ({l2})")

    clip_low, clip_high = clip_bounds(X)
    X_clipped = [clip_row(row, clip_low, clip_high) for row in X]
    X_std, means, stds = standardize(X_clipped)
    weights = fit_logistic(X_std, y, l2=l2)
    train_accuracy = accuracy(weights, X_std, y)

    output = {
        "bot":            bot_name,
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "n_samples":      len(X),
        "features":       feature_names,
        "l2":             l2,
        "oos_accuracy":   round(oos_accuracy, 4) if oos_accuracy is not None else None,
        "clip_low":       clip_low,
        "clip_high":      clip_high,
        "means":          means,
        "stds":           stds,
        "weights":        weights,   # [intercepto, w1, w2, ...] sobre features estandarizadas
        "train_accuracy": round(train_accuracy, 4),
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Modelo reajustado con {len(X)} muestras (L2={l2}) → {weights_path}")
    oos_txt = f" | fuera de muestra (holdout): {oos_accuracy*100:.1f}%" if oos_accuracy is not None else ""
    print(f"     Precisión entrenamiento: {train_accuracy*100:.1f}%{oos_txt}")


# ─────────────────────────────────────────────────────
# CALIBRACIÓN DE ESPORTS (ver docstring del módulo)
# ─────────────────────────────────────────────────────

def build_dataset_esports(log_entries: list[dict], settlements: list[dict]) -> tuple[list[list[float]], list[int], list[str]]:
    """
    A diferencia de build_dataset() (que reconstruye features técnicas para
    reentrenar pesos), aquí la única "feature" es logit(nuestra probabilidad
    cruda del lado apostado) -- lo que se calibra es la confianza del propio
    modelo, no sus insumos.
    """
    settled_index = {
        (s.get("market_id"), s.get("timestamp")): s
        for s in settlements if s.get("bot") == "esports"
    }
    X, y, ts = [], [], []
    for e in log_entries:
        if e.get("action") != "BET":
            continue
        key = (e.get("market_id"), e.get("timestamp"))
        s = settled_index.get(key)
        if s is None:
            continue
        p_equipo_a = e.get("our_prob")
        side = e.get("side")
        if p_equipo_a is None or side not in ("YES", "NO"):
            continue
        # our_prob siempre es P(equipo_a) sin importar el lado apostado --
        # hay que convertirla a P(lado que realmente se jugó).
        p_lado = p_equipo_a if side == "YES" else (1 - p_equipo_a)
        X.append([logit(p_lado)])
        y.append(1 if s.get("won") else 0)
        ts.append(e.get("timestamp") or "")
    return X, y, ts


def retrain_esports_calibration(log_path: str = "esports_bot_log.jsonl",
                                 weights_path: str = "esports_calibration.json",
                                 settlements: list[dict] | None = None):
    print(f"\n── esports (calibración Platt) ({log_path}) ──")

    if ya_reentrenado_recientemente(weights_path):
        print(f"  ⏳ Ya se reentrenó hace menos de {MIN_RETRAIN_INTERVAL_HOURS}h — "
              f"se omite este ciclo para no acumular ruido.")
        return

    log_entries = load_jsonl(log_path)
    X, y, ts = build_dataset_esports(log_entries, settlements or [])

    print(f"  Muestras liquidadas disponibles: {len(X)}")
    if len(X) < ESPORTS_MIN_SAMPLES:
        print(f"  ⚪ Todavía no hay suficientes ({ESPORTS_MIN_SAMPLES} mínimo) — "
              f"sigue con la probabilidad cruda sin calibrar.")
        return

    l2, oos_accuracy = elegir_l2(X, y, ts)
    if oos_accuracy is not None:
        print(f"  🔎 L2 elegido por validación fuera de muestra: {l2} "
              f"(precisión holdout: {oos_accuracy*100:.1f}%)")
    else:
        print(f"  ⚠️  Muestra insuficiente para separar un holdout confiable — "
              f"usando L2 conservador ({l2})")

    clip_low, clip_high = clip_bounds(X)
    X_clipped = [clip_row(row, clip_low, clip_high) for row in X]
    X_std, means, stds = standardize(X_clipped)
    weights = fit_logistic(X_std, y, l2=l2)
    train_accuracy = accuracy(weights, X_std, y)

    output = {
        "calibrador":     "esports",
        "trained_at":     datetime.now(timezone.utc).isoformat(),
        "n_samples":      len(X),
        "feature":        "logit(p_cruda_lado_apostado)",
        "l2":             l2,
        "oos_accuracy":   round(oos_accuracy, 4) if oos_accuracy is not None else None,
        "clip_low":       clip_low,
        "clip_high":      clip_high,
        "means":          means,
        "stds":           stds,
        "weights":        weights,   # [intercepto, pendiente]
        "train_accuracy": round(train_accuracy, 4),
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✅ Calibración reajustada con {len(X)} muestras (L2={l2}) → {weights_path}")
    oos_txt = f" | fuera de muestra (holdout): {oos_accuracy*100:.1f}%" if oos_accuracy is not None else ""
    print(f"     Precisión entrenamiento: {train_accuracy*100:.1f}%{oos_txt}")


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
        historical_fn=build_historical_windows_bot3,
    )
    retrain_esports_calibration(
        log_path="esports_bot_log.jsonl", weights_path="esports_calibration.json",
        settlements=settlements,
    )

    print("\n  ✅ Recalibración completada.")


if __name__ == "__main__":
    main()
