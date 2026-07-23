#!/usr/bin/env python3
"""
=======================================================
  POLYMARKET BTC SCALP BOT (NATIVO A 15 MINUTOS)
  Señal calculada directamente a escala de minutos, no
  reciclando indicadores de 1 hora — bot independiente,
  log y liquidación propios, para comparar su rentabilidad
  real contra btc_direction_bot.py antes de decidir cuál
  (si alguno) merece pasar a dinero real.
=======================================================

POR QUÉ EXISTE:
  btc_direction_bot.py calcula su señal sobre velas de 1 HORA
  pero apuesta en un mercado que resuelve en 15 minutos — un
  desajuste de horizonte real (ver su docstring). Este bot usa
  una señal pensada específicamente para 15 minutos:

  1. Velas de 1 MINUTO de Binance (no de 1 hora)
  2. RSI(14) y cruce EMA(5)/EMA(15) sobre esas velas de 1 min
  3. "Momentum de ventana": cuánto se ha movido BTC desde que
     abrió la ventana de 15 min actual — es la señal más
     directamente relevante, porque es literalmente la pregunta
     que resuelve el mercado (¿subió el precio respecto a como
     abrió esta ventana?)
  4. Combina las tres señales en una probabilidad de UP, la
     compara contra el precio de Polymarket, y apuesta al lado
     con edge real (mismo criterio que btc_direction_bot.py)

⚠️  HIPÓTESIS SIN VALIDAR:
  El término de "momentum de ventana" asume continuación (si ya
  subió, sigue subiendo) — no reversión. Es una apuesta de diseño
  razonable pero NO comprobada; la sección de P&L del reporte
  diario es la que realmente va a decir si esta hipótesis vale
  algo. No hay ninguna garantía de que este modelo tenga más edge
  real que btc_direction_bot.py — por eso corren en paralelo,
  cada uno con su propio log y su propia curva de capital.

⚠️  UNA SOLA APUESTA POR VENTANA:
  A diferencia de correr btc_direction_bot.py con más frecuencia
  (que repetiría la misma opinión horaria varias veces sobre
  distintos precios), este bot SÍ se beneficia de correr seguido
  dentro de la misma ventana de 15 min, porque cada corrida ve
  más momentum acumulado real. Pero para no apilar apuestas
  correlacionadas sobre el mismo mercado, se salta el ciclo si ya
  hay una apuesta registrada sobre ese mismo market_id.

USO:
  python btc_scalp_bot.py              ← 1 ciclo
  python btc_scalp_bot.py --loop       ← corre cada 5 min indefinidamente
"""

import json
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

from btc_direction_bot import (
    fetch, ema, rsi,
    parse_window_minutes, find_btc_window_market, get_market_probability,
    BINANCE_URL, SYMBOL,
)

# En consolas Windows con codepage legado (cp1252), imprimir emojis revienta
# con UnicodeEncodeError. Forzamos stdout/stderr a UTF-8 si el terminal lo permite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CANDLES_1M   = 30          # velas de 1 minuto para RSI/EMA de corto plazo
EDGE_MINIMO  = 0.10        # ventaja mínima para apostar (10% — más exigente que
                           # btc_direction_bot.py porque la señal es más ruidosa)
APUESTA_USD  = 10.0        # tamaño fijo de apuesta en simulación
LOG_FILE     = "btc_scalp_log.jsonl"


# ─────────────────────────────────────────────────────
# DATOS DE CORTO PLAZO
# ─────────────────────────────────────────────────────

def get_1m_candles(limit: int = CANDLES_1M) -> list[dict]:
    """Últimas velas de 1 minuto de BTC/USDT."""
    url = f"{BINANCE_URL}/api/v3/klines?symbol={SYMBOL}&interval=1m&limit={limit}"
    raw = fetch(url)
    return [{"open_time": c[0], "open": float(c[1]), "close": float(c[4])} for c in raw]


def get_window_open_price(market: dict, duration_minutes: float) -> float | None:
    """
    Precio de BTC en el momento en que abrió la ventana de 15 min actual,
    para medir cuánto se ha movido ya el precio dentro de esta ventana.
    """
    try:
        end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
        window_start = end - timedelta(minutes=duration_minutes)
        start_ms = int(window_start.timestamp() * 1000) - 60_000   # 1 min de margen
        end_ms   = start_ms + 5 * 60_000
        url = (f"{BINANCE_URL}/api/v3/klines?symbol={SYMBOL}&interval=1m"
               f"&startTime={start_ms}&endTime={end_ms}&limit=5")
        raw = fetch(url)
        if not raw:
            return None
        return float(raw[0][1])   # precio de apertura de la primera vela en el rango
    except Exception:
        return None


# ─────────────────────────────────────────────────────
# ANÁLISIS DE CORTO PLAZO
# ─────────────────────────────────────────────────────

def analyze_scalp(candles_1m: list[dict], window_open_price: float | None,
                   current_price: float) -> dict:
    closes = [c["close"] for c in candles_1m]

    rsi_val = rsi(closes)
    ema5    = ema(closes, 5)
    ema15   = ema(closes, 15)
    ema_diff = ema5[-1] - ema15[-1]

    window_momentum = None
    if window_open_price:
        window_momentum = (current_price - window_open_price) / window_open_price * 100

    return {
        "rsi":              rsi_val,
        "ema_diff":         round(ema_diff, 2),
        "window_momentum":  round(window_momentum, 4) if window_momentum is not None else None,
        "btc_price":        current_price,
    }


def our_probability_scalp(analysis: dict) -> float:
    """
    Estima la probabilidad de que BTC termine arriba al cierre de la ventana.
    Base 50% + RSI de corto plazo (contrarian) + cruce EMA rápido (tendencia)
    + momentum ya recorrido dentro de la ventana (continuación — ver aviso
    de hipótesis sin validar en el docstring del módulo).
    """
    base = 0.50

    rsi_val = analysis["rsi"]
    if rsi_val < 25:      base += 0.08
    elif rsi_val < 35:    base += 0.04
    elif rsi_val > 75:    base -= 0.08
    elif rsi_val > 65:    base -= 0.04

    if analysis["ema_diff"] > 0:
        base += 0.03
    else:
        base -= 0.03

    wm = analysis["window_momentum"]
    if wm is not None:
        # Continuación: cada 0.1% ya movido dentro de la ventana desplaza la
        # probabilidad, con un tope de +-15 puntos para no sobreconfiar en
        # un movimiento que podría revertir en lo que queda de ventana.
        base += max(-0.15, min(0.15, wm * 0.5))

    return round(max(0.10, min(0.90, base)), 4)


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


def already_bet_on(market_id, entries: list[dict]) -> bool:
    """Evita apilar más de una apuesta sobre la misma ventana de 15 min."""
    return any(e.get("market_id") == market_id and e.get("action") == "BET" for e in entries)


# ─────────────────────────────────────────────────────
# CICLO PRINCIPAL
# ─────────────────────────────────────────────────────

def run_cycle() -> dict:
    now = datetime.now(timezone.utc)
    print(f"\n{'─'*60}")
    print(f"  🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─'*60}")

    market = find_btc_window_market()
    if not market:
        print("  ⚠️  No hay ventana de 15 min activa ahora mismo")
        entry = {"timestamp": now.isoformat(), "action": "NO_MARKET"}
        log_entry(entry)
        return entry

    market_id = market.get("id")
    print(f"  🏪 {market.get('question','')[:70]}")

    existing = load_log()
    if already_bet_on(market_id, existing):
        print("  ⏭️  Ya hay una apuesta registrada sobre esta ventana — no se repite")
        entry = {"timestamp": now.isoformat(), "action": "SKIP", "reason": "YA_APOSTADO",
                  "market_id": market_id}
        log_entry(entry)
        return entry

    try:
        candles = get_1m_candles()
        print(f"  📊 {len(candles)} velas 1min descargadas de Binance")
    except Exception as e:
        print(f"  ❌ Error descargando velas: {e}")
        entry = {"timestamp": now.isoformat(), "action": "ERROR", "reason": str(e)}
        log_entry(entry)
        return entry

    current_price = candles[-1]["close"]
    duration = parse_window_minutes(market.get("question", "")) or 15
    window_open_price = get_window_open_price(market, duration)

    analysis = analyze_scalp(candles, window_open_price, current_price)
    print(f"\n  💰 BTC/USDT: ${current_price:,.2f}")
    print(f"     RSI(14) 1min:      {analysis['rsi']:.1f}")
    print(f"     EMA 5/15 diff:     {analysis['ema_diff']:+.2f}")
    if analysis["window_momentum"] is not None:
        print(f"     Momentum ventana:  {analysis['window_momentum']:+.3f}%")
    else:
        print(f"     Momentum ventana:  sin dato (no se pudo ubicar la vela de apertura)")

    our_prob_up = our_probability_scalp(analysis)
    market_prob_up = get_market_probability(market, "UP")
    print(f"\n  🧮 Nuestra probabilidad de UP: {our_prob_up*100:.1f}%")
    print(f"     Precio mercado (UP): {market_prob_up*100:.1f}¢")

    edge_up = our_prob_up - market_prob_up
    if edge_up >= 0:
        bet_side, our_prob, market_prob, edge = "UP", our_prob_up, market_prob_up, edge_up
    else:
        bet_side, our_prob, market_prob, edge = "DOWN", 1 - our_prob_up, 1 - market_prob_up, -edge_up

    print(f"     Edge ({bet_side}): {edge*100:.1f}%")

    if edge >= EDGE_MINIMO:
        action = "BET"
        print(f"\n  ✅ APUESTA SIMULADA: {bet_side} ${APUESTA_USD:.2f}  (edge={edge*100:.1f}%)")
    else:
        action = "PASS"
        print(f"\n  ⚪ Edge insuficiente ({edge*100:.1f}% < {EDGE_MINIMO*100:.0f}%) — no se apuesta")

    entry = {
        "timestamp":       now.isoformat(),
        "action":          action,
        "bet_side":        bet_side if action == "BET" else None,
        "bet_usd":         APUESTA_USD if action == "BET" else 0,
        "our_prob":        our_prob,
        "market_prob":     market_prob,
        "edge":            round(edge, 4),
        "btc_price":       current_price,
        "rsi":             analysis["rsi"],
        "ema_diff":        analysis["ema_diff"],
        "window_momentum": analysis["window_momentum"],
        "market_id":       market_id,
        "market_q":        market.get("question", "")[:80],
    }
    log_entry(entry)
    return entry


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC Scalp Bot — señal nativa de 15 min")
    parser.add_argument("--loop", action="store_true",
                        help="Corre indefinidamente cada 5 minutos")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 BTC SCALP BOT — VENTANA DE 15 MIN")
    print("=" * 60)
    print(f"  Fuente: Binance {SYMBOL} 1min")
    print(f"  Señal: RSI/EMA de 1min + momentum de ventana")
    print(f"  Edge mínimo: {EDGE_MINIMO*100:.0f}%")
    print(f"  Apuesta simulada: ${APUESTA_USD:.2f} por señal (máx. 1 por ventana)")
    print(f"  Log: {LOG_FILE}")
    print(f"  Modo: {'LOOP (cada 5 min)' if args.loop else '1 CICLO'}")

    if args.loop:
        print("\n  Presiona Ctrl+C para detener\n")
        try:
            while True:
                run_cycle()
                time.sleep(5 * 60)
        except KeyboardInterrupt:
            print("\n\n  🛑 Bot detenido.")
    else:
        result = run_cycle()
        print(f"\n  💡 Para correr en loop: python btc_scalp_bot.py --loop")
        print(f"  💡 Para GitHub Actions: usa el workflow btc-scalp.yml\n")
        if result.get("action") == "ERROR":
            sys.exit(1)


if __name__ == "__main__":
    main()
