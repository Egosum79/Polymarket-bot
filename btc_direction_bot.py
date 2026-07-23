#!/usr/bin/env python3
"""
=======================================================
  POLYMARKET BTC DIRECTION BOT (VENTANA DE 15 MIN)
  Predice si BTC sube o baja y apuesta en el mercado
  "BTC Up or Down" de 15 minutos que esté abierto
=======================================================

CÓMO FUNCIONA:
  1. Descarga las últimas 50 velas 1H de BTC/USDT desde Binance
  2. Calcula RSI(14), cruce EMA9/EMA21, MACD
  3. Si 2 de 3 indicadores coinciden → señal fuerte UP o DOWN
  4. Busca el mercado "BTC Up or Down" de 15 minutos actualmente
     abierto en Polymarket (Polymarket ya no ofrece la variante de
     1 hora — solo existen ventanas de 5 y 15 minutos, verificado
     en vivo contra la API el 2026-07-22)
  5. Si hay ventaja ≥ 6% → registra apuesta simulada

⚠️  DESAJUSTE DE HORIZONTE (limitación conocida):
  Los indicadores (RSI/EMA/MACD) se calculan sobre velas de 1 HORA
  y están pensados para predecir la tendencia de la próxima hora,
  no de los próximos 15 minutos. En una ventana tan corta domina el
  ruido de corto plazo, así que la señal probablemente tenga menos
  poder predictivo real del que el modelo sugiere. Se mantiene así
  porque es la única cadencia horaria de mercado que queda
  disponible en Polymarket para BTC — pero conviene verificar la
  rentabilidad real (ver settle_bets.py / sección P&L del reporte
  diario) antes de considerar apostar dinero real con este bot.

FUENTE DE DATOS:
  Binance BTC/USDT (misma fuente que usa Polymarket para resolver)

USO:
  python btc_direction_bot.py              ← 1 ciclo
  python btc_direction_bot.py --loop       ← corre cada hora indefinidamente

MODO REAL (futuro):
  Requiere py-clob-client + credenciales de Polymarket
"""

import json
import math
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# En consolas Windows con codepage legado (cp1252), imprimir emojis revienta
# con UnicodeEncodeError. Forzamos stdout/stderr a UTF-8 si el terminal lo permite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

# data-api.binance.vision es el espejo público de solo-datos-de-mercado de Binance:
# mismo formato que api.binance.com, pero sin el bloqueo geográfico (HTTP 451) que
# afecta a api.binance.com desde IPs de datacenter en EE.UU. (incluyendo runners
# de GitHub Actions).
BINANCE_URL   = "https://data-api.binance.vision"
GAMMA_URL     = "https://gamma-api.polymarket.com"

SYMBOL        = "BTCUSDT"
INTERVAL      = "1h"
CANDLES       = 50          # velas históricas para indicadores

EDGE_MINIMO   = 0.06        # ventaja mínima para apostar (6%)
APUESTA_USD   = 10.0        # tamaño fijo de apuesta en simulación
LOG_FILE      = "btc_bot_log.jsonl"

# ─────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────

def fetch(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_candles(symbol: str = SYMBOL, interval: str = INTERVAL, limit: int = CANDLES) -> list[dict]:
    """Descarga velas OHLCV de Binance."""
    url = f"{BINANCE_URL}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    raw = fetch(url)
    candles = []
    for c in raw:
        candles.append({
            "open_time": c[0],
            "open":      float(c[1]),
            "high":      float(c[2]),
            "low":       float(c[3]),
            "close":     float(c[4]),
            "volume":    float(c[5]),
        })
    return candles


# ─────────────────────────────────────────────────────
# INDICADORES TÉCNICOS (sin librerías externas)
# ─────────────────────────────────────────────────────

def ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def rsi(closes: list[float], period: int = 14) -> float:
    """RSI — Relative Strength Index."""
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    if len(gains) < period:
        return 50.0

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def macd(closes: list[float]) -> dict:
    """MACD — línea MACD, señal y histograma."""
    ema12  = ema(closes, 12)
    ema26  = ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal    = ema(macd_line, 9)
    histogram = [macd_line[i] - signal[i] for i in range(len(macd_line))]
    return {
        "macd":      macd_line[-1],
        "signal":    signal[-1],
        "histogram": histogram[-1],
        "prev_hist": histogram[-2] if len(histogram) > 1 else 0,
    }


def analyze_candles(candles: list[dict]) -> dict:
    """
    Calcula todos los indicadores y genera señal compuesta.
    Retorna señal: 'UP', 'DOWN' o 'NEUTRAL'
    """
    closes  = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # ── RSI ──────────────────────────────────────────
    rsi_val    = rsi(closes)
    rsi_signal = "UP" if rsi_val < 40 else "DOWN" if rsi_val > 60 else "NEUTRAL"

    # ── EMA 9/21 cruce ───────────────────────────────
    ema9       = ema(closes, 9)
    ema21      = ema(closes, 21)
    ema_diff   = ema9[-1] - ema21[-1]
    ema_signal = "UP" if ema_diff > 0 else "DOWN"

    # ── MACD ─────────────────────────────────────────
    macd_data  = macd(closes)
    # Señal: si el histograma cruzó de negativo a positivo (o viceversa)
    if macd_data["histogram"] > 0 and macd_data["prev_hist"] <= 0:
        macd_signal = "UP"
    elif macd_data["histogram"] < 0 and macd_data["prev_hist"] >= 0:
        macd_signal = "DOWN"
    elif macd_data["histogram"] > 0:
        macd_signal = "UP"
    else:
        macd_signal = "DOWN"

    # ── Volumen (confirmación) ────────────────────────
    vol_avg     = sum(volumes[-10:]) / 10
    vol_current = volumes[-1]
    vol_surge   = vol_current > vol_avg * 1.3   # volumen 30% sobre promedio

    # ── Señal compuesta ──────────────────────────────
    signals = [rsi_signal, ema_signal, macd_signal]
    up_count   = signals.count("UP")
    down_count = signals.count("DOWN")

    if up_count >= 2:
        direction = "UP"
        strength  = up_count / 3
    elif down_count >= 2:
        direction = "DOWN"
        strength  = down_count / 3
    else:
        direction = "NEUTRAL"
        strength  = 0.0

    # Fuerza extra si el volumen confirma
    if vol_surge and direction != "NEUTRAL":
        strength = min(strength + 0.1, 1.0)

    # Precio actual
    current_price = closes[-1]
    prev_close    = closes[-2]
    momentum_1h   = (current_price - prev_close) / prev_close * 100

    return {
        "direction":    direction,
        "strength":     round(strength, 3),
        "rsi":          rsi_val,
        "rsi_signal":   rsi_signal,
        "ema_diff":     round(ema_diff, 2),
        "ema_signal":   ema_signal,
        "macd_hist":    round(macd_data["histogram"], 2),
        "macd_signal":  macd_signal,
        "vol_surge":    vol_surge,
        "btc_price":    current_price,
        "momentum_1h":  round(momentum_1h, 3),
    }


# ─────────────────────────────────────────────────────
# BUSCAR MERCADO ACTIVO EN POLYMARKET
# ─────────────────────────────────────────────────────

def parse_window_minutes(question: str) -> float | None:
    """
    Extrae la duración real de la ventana desde el propio texto de la
    pregunta (ej. "...2:45PM-3:00PM ET" → 15.0 minutos).

    Nota: el campo 'startDate' de la API NO sirve para esto — se
    verificó en vivo que para estos mercados 'startDate' refleja
    cuándo Polymarket *creó/listó* el mercado (típicamente ~1 día
    antes), no cuándo abre la ventana de apuesta. El rango horario
    solo está confiablemente disponible en el texto de la pregunta.
    """
    times = re.findall(r'(\d{1,2}:\d{2}\s*[AP]M)', question, re.IGNORECASE)
    if len(times) < 2:
        return None
    try:
        t0 = datetime.strptime(times[0].replace(" ", "").upper(), "%I:%M%p")
        t1 = datetime.strptime(times[1].replace(" ", "").upper(), "%I:%M%p")
        diff = (t1 - t0).total_seconds() / 60
        if diff < 0:
            diff += 24 * 60
        return diff
    except Exception:
        return None


def find_btc_window_market() -> dict | None:
    """
    Busca el mercado 'BTC Up or Down' de 15 minutos actualmente abierto en
    Polymarket. (Se verificó en vivo el 2026-07-22 que Polymarket ya no
    ofrece la variante de 1 hora para BTC — solo existen ventanas de 5 y
    15 minutos; ver el aviso de desajuste de horizonte en el docstring del
    módulo.)
    Filtra por: mercado activo, resuelve pronto, ventana ≈15 min, precio
    entre 30-70¢.
    """
    now = datetime.now(timezone.utc)
    # Duración esperada de la ventana (tolerancia +-2 min sobre 15)
    WINDOW_MIN_OK = 13
    WINDOW_MAX_OK = 17
    # No mirar más allá del propio largo de la ventana + margen: si buscáramos
    # más lejos podríamos enganchar una ventana de 15 min que abre en 2 horas
    # en vez de la que está abierta ahora mismo.
    MAX_MINUTES_LEFT = 20
    later = now + timedelta(minutes=MAX_MINUTES_LEFT)

    try:
        # Filtrar por end_date_min/max directamente en la API (en vez de pedir
        # "las N más próximas a vencer" y filtrar en el cliente) evita por
        # completo los mercados "zombis": entradas marcadas active=true &
        # closed=false que en realidad vencieron hace meses y nunca se
        # cerraron formalmente. Sin este filtro de rango, esos zombis dominan
        # cualquier orden por endDate y tapan los mercados reales.
        url = (f"{GAMMA_URL}/markets?limit=100&active=true&closed=false"
               f"&end_date_min={now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
               f"&end_date_max={later.strftime('%Y-%m-%dT%H:%M:%SZ')}"
               f"&order=endDate&ascending=true")
        markets = fetch(url)
    except Exception as e:
        print(f"  ⚠️  Error buscando mercados: {e}")
        return None

    hour_keywords = ["up or down", "arriba o abajo"]
    btc_keywords  = ["bitcoin", "btc"]

    best = None
    best_minutes = 999

    for m in markets:
        question = m.get("question", "")
        q = question.lower()

        # Debe ser mercado de BTC
        if not any(kw in q for kw in btc_keywords):
            continue
        # Debe ser Up or Down
        if not any(kw in q for kw in hour_keywords):
            continue

        # Verificar que resuelva pronto (la ventana actualmente abierta)
        end_date = m.get("endDate", "")
        try:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            minutes_left = (end - now).total_seconds() / 60
            if minutes_left < 0 or minutes_left > MAX_MINUTES_LEFT:
                continue
        except Exception:
            continue

        # Quedarnos solo con la ventana de 15 min (no la de 5 min), leyendo
        # el rango horario directamente del texto de la pregunta.
        duration_minutes = parse_window_minutes(question)
        if duration_minutes is None or not (WINDOW_MIN_OK <= duration_minutes <= WINDOW_MAX_OK):
            continue

        # Excluir solo mercados prácticamente ya resueltos. En una ventana de
        # 15 min el precio se mueve rápido hacia el lado que va ganando
        # apenas empieza — exigir 30-70¢ (como si acabara de abrir) descartaba
        # casi todos los mercados reales que se alcanzan a atrapar con una
        # revisión cada hora. 4-96¢ (mismo criterio que usa polymarket_bot.py)
        # solo filtra los que ya están prácticamente decididos.
        try:
            prices    = json.loads(m.get("outcomePrices", "[0.5,0.5]"))
            yes_price = float(prices[0])
            if not (0.04 < yes_price < 0.96):
                continue
        except Exception:
            continue

        # Tomar el que vence más pronto
        if minutes_left < best_minutes:
            best_minutes = minutes_left
            best = m

    if best:
        print(f"     ⏱️  Vence en {best_minutes:.0f} minutos")
    return best


def get_market_probability(market: dict, direction: str) -> float:
    """Retorna la probabilidad implícita del lado que queremos apostar."""
    try:
        prices    = json.loads(market.get("outcomePrices", "[0.5,0.5]"))
        up_price  = float(prices[0])   # YES = UP en estos mercados
        down_price = float(prices[1])  # NO  = DOWN
        return up_price if direction == "UP" else down_price
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────
# ESTIMACIÓN DE PROBABILIDAD PROPIA
# ─────────────────────────────────────────────────────

WEIGHTS_FILE = "bot2_weights.json"


def _predict_learned(analysis: dict, weights_path: str = WEIGHTS_FILE) -> float | None:
    """
    Si retrain_model.py ya ajustó un modelo con suficientes apuestas reales
    liquidadas, usa esos pesos en vez de la heurística fija de abajo.
    Retorna None si el archivo no existe todavía (bots siguen con la
    heurística original hasta entonces) o si algo en él no es válido.
    """
    p = Path(weights_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            model = json.load(f)
        means, stds, weights = model["means"], model["stds"], model["weights"]
        ema_signal  = 1.0 if analysis["ema_signal"] == "UP" else -1.0
        macd_signal = 1.0 if analysis["macd_signal"] == "UP" else -1.0
        raw = [analysis["rsi"], ema_signal, macd_signal, analysis["momentum_1h"]]
        std = [(x - m) / s for x, m, s in zip(raw, means, stds)]
        z = weights[0] + sum(w * x for w, x in zip(weights[1:], std))
        prob = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
        return round(max(0.02, min(0.98, prob)), 4)
    except Exception:
        return None


def our_probability(analysis: dict) -> float:
    """
    Estima nuestra probabilidad de que BTC suba en la próxima hora.
    Usa el modelo reajustado con datos reales si ya existe (ver
    retrain_model.py); si no, cae en la heurística fija de siempre:
    base 50% (mercado eficiente) + ajustes por indicadores.
    """
    learned = _predict_learned(analysis)
    if learned is not None:
        return learned

    base = 0.50

    # RSI: sobreventa → más probable subida
    rsi_val = analysis["rsi"]
    if rsi_val < 30:
        base += 0.10
    elif rsi_val < 40:
        base += 0.05
    elif rsi_val > 70:
        base -= 0.10
    elif rsi_val > 60:
        base -= 0.05

    # EMA: tendencia
    if analysis["ema_signal"] == "UP":
        base += 0.05
    else:
        base -= 0.05

    # MACD: momento
    if analysis["macd_signal"] == "UP":
        base += 0.04
    else:
        base -= 0.04

    # Volumen confirma movimiento
    if analysis["vol_surge"]:
        # El volumen amplifica la señal existente
        direction_mult = 1 if base > 0.5 else -1
        base += direction_mult * 0.03

    # Momentum reciente de 1h
    mom = analysis["momentum_1h"]
    if mom > 0.5:    base += 0.03
    elif mom < -0.5: base -= 0.03

    return round(max(0.15, min(0.85, base)), 4)


# ─────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────

def log_entry(entry: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_log() -> list[dict]:
    p = Path(LOG_FILE)
    if not p.exists():
        return []
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def print_stats():
    """Muestra estadísticas acumuladas del log."""
    entries = load_log()
    if not entries:
        return
    bets = [e for e in entries if e.get("action") == "BET"]
    if not bets:
        print(f"\n  📋 {len(entries)} ciclos registrados, sin apuestas aún.")
        return
    print(f"\n  📋 ESTADÍSTICAS ACUMULADAS")
    print(f"     Ciclos totales: {len(entries)}")
    print(f"     Apuestas simuladas: {len(bets)}")
    up_bets   = [b for b in bets if b.get("bet_side") == "UP"]
    down_bets = [b for b in bets if b.get("bet_side") == "DOWN"]
    print(f"     UP: {len(up_bets)}  |  DOWN: {len(down_bets)}")
    total_sim = sum(b.get("bet_usd", 0) for b in bets)
    print(f"     Capital simulado apostado: ${total_sim:.2f}")


# ─────────────────────────────────────────────────────
# CICLO PRINCIPAL
# ─────────────────────────────────────────────────────

def run_cycle() -> dict:
    now = datetime.now(timezone.utc)
    print(f"\n{'─'*60}")
    print(f"  🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*60}")

    # 1. Descargar velas de Binance
    try:
        candles = get_candles()
        print(f"  📊 {len(candles)} velas 1H descargadas de Binance")
    except Exception as e:
        print(f"  ❌ Error descargando velas: {e}")
        entry = {"timestamp": now.isoformat(), "action": "ERROR", "reason": str(e)}
        log_entry(entry)   # registrar el fallo para que el reporte y el historial lo reflejen
        return entry

    # 2. Analizar indicadores
    analysis = analyze_candles(candles)
    btc = analysis["btc_price"]

    print(f"\n  💰 BTC/USDT: ${btc:,.2f}  |  Momentum 1h: {analysis['momentum_1h']:+.3f}%")
    print(f"\n  📈 INDICADORES:")
    print(f"     RSI(14):     {analysis['rsi']:.1f}  → {analysis['rsi_signal']}")
    print(f"     EMA 9/21:    diff={analysis['ema_diff']:+.2f}  → {analysis['ema_signal']}")
    print(f"     MACD hist:   {analysis['macd_hist']:+.4f}  → {analysis['macd_signal']}")
    print(f"     Vol surge:   {'✅ SÍ' if analysis['vol_surge'] else '❌ NO'}")
    print(f"\n  🎯 SEÑAL COMPUESTA: {analysis['direction']}  (fuerza: {analysis['strength']*100:.0f}%)")

    # 3. Si señal es NEUTRAL → no apostar
    if analysis["direction"] == "NEUTRAL":
        print(f"\n  ⚪ Sin señal clara — no se apuesta este ciclo")
        entry = {
            "timestamp": now.isoformat(),
            "action":    "SKIP",
            "reason":    "NEUTRAL",
            "btc_price": btc,
            **{k: analysis[k] for k in ["rsi","ema_signal","macd_signal","momentum_1h"]},
        }
        log_entry(entry)
        return entry

    # 4. Calcular probabilidad de UP (our_probability siempre devuelve P(sube),
    #    nunca P(bet_side) — hay que compararla contra el precio de mercado
    #    de UP y recién ahí decidir a qué lado le conviene apostar).
    our_prob_up = our_probability(analysis)
    print(f"\n  🧮 Nuestra probabilidad de UP: {our_prob_up*100:.1f}%  "
          f"(señal técnica: {analysis['direction']})")

    # 5. Buscar mercado activo en Polymarket
    market = find_btc_window_market()
    if market:
        market_prob_up = get_market_probability(market, "UP")
        print(f"  🏪 Mercado Polymarket encontrado:")
        print(f"     {market.get('question','')[:70]}")
        print(f"     Precio mercado (UP): {market_prob_up*100:.1f}¢")
    else:
        market_prob_up = 0.50
        print(f"  ⚠️  No se encontró mercado horario activo — usando base 50/50")

    # 6. Elegir el lado con edge real, sin importar hacia dónde apuntaban los
    #    indicadores: si el mercado ya descuenta con fuerza un lado y nuestro
    #    modelo discrepa, el edge de verdad puede estar en el lado contrario.
    edge_up = our_prob_up - market_prob_up
    if edge_up >= 0:
        bet_side, our_prob, market_prob, edge = "UP", our_prob_up, market_prob_up, edge_up
    else:
        bet_side, our_prob, market_prob, edge = "DOWN", 1 - our_prob_up, 1 - market_prob_up, -edge_up

    print(f"     Edge ({bet_side}): {edge*100:.1f}%")

    # 7. Decisión de apuesta
    if edge >= EDGE_MINIMO:
        action = "BET"
        print(f"\n  ✅ APUESTA SIMULADA: {bet_side} ${APUESTA_USD:.2f}  (edge={edge*100:.1f}%)")
    else:
        action = "PASS"
        print(f"\n  ⚪ Edge insuficiente ({edge*100:.1f}% < {EDGE_MINIMO*100:.0f}%) — no se apuesta")

    entry = {
        "timestamp":    now.isoformat(),
        "action":       action,
        "bet_side":     bet_side if action == "BET" else None,
        "bet_usd":      APUESTA_USD if action == "BET" else 0,
        "our_prob":     our_prob,
        "market_prob":  market_prob,
        "edge":         round(edge, 4),
        "btc_price":    btc,
        "rsi":          analysis["rsi"],
        "ema_signal":   analysis["ema_signal"],
        "macd_signal":  analysis["macd_signal"],
        "strength":     analysis["strength"],
        "momentum_1h":  analysis["momentum_1h"],
        "market_id":    market.get("id") if market else None,
        "market_q":     market.get("question","")[:80] if market else None,
    }
    log_entry(entry)
    return entry


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC Direction Bot — 1H")
    parser.add_argument("--loop", action="store_true",
                        help="Corre indefinidamente cada hora")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 BTC DIRECTION BOT — VELAS 1 HORA")
    print("=" * 60)
    print(f"  Fuente: Binance {SYMBOL} {INTERVAL}")
    print(f"  Indicadores: RSI(14) + EMA(9/21) + MACD")
    print(f"  Edge mínimo: {EDGE_MINIMO*100:.0f}%")
    print(f"  Apuesta simulada: ${APUESTA_USD:.2f} por señal")
    print(f"  Log: {LOG_FILE}")
    print(f"  Modo: {'LOOP (cada hora)' if args.loop else '1 CICLO'}")

    if args.loop:
        print("\n  Presiona Ctrl+C para detener\n")
        try:
            while True:
                run_cycle()
                print_stats()
                # Esperar hasta el inicio del próximo ciclo de hora
                now     = datetime.now(timezone.utc)
                mins_left = 60 - now.minute
                print(f"\n  ⏰ Próximo ciclo en {mins_left} minutos...")
                time.sleep(mins_left * 60)
        except KeyboardInterrupt:
            print("\n\n  🛑 Bot detenido.")
    else:
        result = run_cycle()
        print_stats()
        print(f"\n  💡 Para correr en loop: python btc_direction_bot.py --loop")
        print(f"  💡 Para GitHub Actions: usa el workflow btc-direction.yml\n")
        if result.get("action") == "ERROR":
            # Sin esto, un fallo al descargar datos (ej. Binance bloqueando la IP
            # del runner de GitHub Actions) quedaba en silencio: el workflow
            # marcaba "success" aunque el ciclo no hiciera nada.
            sys.exit(1)


if __name__ == "__main__":
    main()
