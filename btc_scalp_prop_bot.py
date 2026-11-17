#!/usr/bin/env python3
"""
=======================================================
  BOT 5 — BTC SCALP PROPORCIONAL (experimento paralelo a Bot 3)
=======================================================

QUÉ ES:
  Exactamente la misma señal, filtros y modelo que btc_scalp_bot.py (Bot 3)
  -- se importa y se reutiliza run_cycle() tal cual, así que cualquier
  arreglo futuro de Bot 3 se hereda solo y la comparación es limpia. La
  ÚNICA diferencia es el tamaño de la apuesta:

      Bot 3:  $10 fijos por señal.
      Bot 5:  5% del capital simulado de Bot 5, con piso $10 y tope $50.

POR QUÉ EXISTE:
  Análisis 2026-10-09 sobre las 235 apuestas de Bot 3 posteriores al veto
  de momentum (60.9% de acierto, IC 95% 54.5%-66.9%). Simulando tamaños con
  Monte Carlo, ~5% del capital fue el punto que aguantaba todos los
  escenarios de ventaja (observada, 56%, 52% y nula) sin que las caídas
  explotaran; 10-20% multiplicaba el resultado si la ventaja era real pero
  destruía el capital si se debilitaba. Este bot prueba esa regla en vivo,
  en paralelo, sin tocar a Bot 3.

CÓMO SE CALCULA LA APUESTA:
  capital = CAPITAL_INICIAL + suma del pnl de las apuestas YA LIQUIDADAS de
  este bot (settlements.jsonl, bot == "bot5"). apuesta = 5% del capital,
  acotada a [$10, $50]. Como settle_bets.py corre una vez al día, el
  capital (y por tanto la apuesta) se actualiza una vez al día, no apuesta
  por apuesta -- es deliberado: evita reaccionar a rachas de pocas
  apuestas. Arranca en $100, así que se comporta igual que Bot 3 ($10)
  hasta que el capital simulado supere $200, y alcanza el tope de $50 al
  llegar a $1,000.

MEDICIÓN DE LIQUIDEZ (no afecta decisiones):
  Cada vez que Bot 5 apuesta, snapshot_libro() lee el libro de órdenes del
  lado apostado (CLOB) y guarda en btc_scalp_prop_book.jsonl: mejor bid/ask,
  spread, profundidad en USD a 1/3/5 centavos del mejor ask, y el precio
  promedio de llenado (vwap) y deslizamiento para la apuesta real, $50 y
  $100. Sirve para decidir con datos si el tope de $50 es realista. Si el
  CLOB no responde desde el runner, queda {"error": ...} en el log.

LÍMITES CONOCIDOS:
  - No modela comisiones ni deslizamiento (el tope de $50 es provisional
    hasta medir la profundidad real del libro de estas ventanas de 15 min).
  - Con capital < $200 el piso de $10 domina (no es 5%).
  - Es simulación: no ejecuta órdenes reales.

USO:
  python btc_scalp_prop_bot.py            ← un ciclo
  python btc_scalp_prop_bot.py --loop     ← cada 5 minutos
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import btc_scalp_bot as base

STAKE_PCT        = 0.05    # fracción del capital simulado por apuesta
STAKE_MIN        = 10.0    # piso (igual al tamaño fijo de Bot 3)
STAKE_MAX        = 50.0    # tope provisional hasta medir liquidez real
CAPITAL_INICIAL  = 100.0   # mismo banco de arranque que el resto de los bots
BOT_TAG          = "bot5"  # etiqueta en settlements.jsonl (ver settle_bets.py)
LOG_FILE_PROP    = "btc_scalp_prop_log.jsonl"
SETTLEMENTS_FILE = "settlements.jsonl"
BOOK_LOG_FILE    = "btc_scalp_prop_book.jsonl"   # profundidad real del libro en cada apuesta (medición, no afecta decisiones)
GAMMA_URL        = "https://gamma-api.polymarket.com"
CLOB_URL         = "https://clob.polymarket.com"


def capital_simulado() -> float:
    capital = CAPITAL_INICIAL
    p = Path(SETTLEMENTS_FILE)
    if not p.exists():
        return capital
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except Exception:
                continue
            if s.get("bot") == BOT_TAG:
                capital += s.get("pnl", 0) or 0
    return capital


def calcular_apuesta(capital: float) -> float:
    return round(min(STAKE_MAX, max(STAKE_MIN, STAKE_PCT * capital)), 2)


def _get_json(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def analizar_libro(asks: list[tuple[float, float]], bids: list[tuple[float, float]],
                   montos: list[float]) -> dict:
    """
    Dado el libro (listas de (precio, shares)), calcula cuánto costaría COMPRAR
    cada monto en USD recorriendo los asks de menor a mayor precio.

    Por monto: precio promedio de llenado (vwap), deslizamiento contra el mejor
    ask en centavos, y fracción realmente llenada (<1 = el libro no alcanza).
    También: profundidad en USD a <=1, 3 y 5 centavos del mejor ask.
    """
    asks = sorted(asks)
    bids = sorted(bids, reverse=True)
    out = {
        "best_ask": asks[0][0] if asks else None,
        "best_bid": bids[0][0] if bids else None,
        "ask_levels": len(asks),
    }
    if out["best_ask"] is not None and out["best_bid"] is not None:
        out["spread"] = round(out["best_ask"] - out["best_bid"], 4)
    if not asks:
        return out
    best = asks[0][0]
    out["depth_usd"] = {
        f"{c}c": round(sum(p * s for p, s in asks if p <= best + c / 100 + 1e-9), 2)
        for c in (1, 3, 5)
    }
    fills = {}
    for monto in montos:
        restante, gastado, shares = monto, 0.0, 0.0
        for p, s in asks:
            if restante <= 1e-9:
                break
            usd_nivel = p * s
            tomar = min(restante, usd_nivel)
            gastado += tomar
            shares += tomar / p
            restante -= tomar
        llenado = gastado / monto if monto else 0.0
        vwap = gastado / shares if shares else None
        fills[f"{monto:g}"] = {
            "vwap": round(vwap, 4) if vwap else None,
            "slippage_c": round((vwap - best) * 100, 2) if vwap else None,
            "llenado": round(llenado, 3),
        }
    out["fills"] = fills
    return out


def snapshot_libro(entry: dict) -> None:
    """
    Solo tras una apuesta: guarda la profundidad REAL del libro del lado
    apostado (para decidir con datos si el tope de $50 es realista). Nunca
    debe romper el ciclo -- cualquier fallo se registra como error y ya.
    Si el CLOB no responde desde el runner, el log lo mostrará (queda
    {"error": ...}), lo cual ya es información útil.
    """
    reg = {
        "timestamp":  entry.get("timestamp"),
        "market_id":  entry.get("market_id"),
        "bet_side":   entry.get("bet_side"),
        "bet_usd":    entry.get("bet_usd"),
        "market_prob": entry.get("market_prob"),
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        mk = _get_json(f"{GAMMA_URL}/markets/{entry['market_id']}")
        tokens = json.loads(mk["clobTokenIds"])
        outcomes = [str(o).lower() for o in json.loads(mk["outcomes"])]
        token = tokens[outcomes.index("up" if entry["bet_side"] == "UP" else "down")]
        reg["gamma_best_bid"] = mk.get("bestBid")
        reg["gamma_best_ask"] = mk.get("bestAsk")
        reg["gamma_liquidity"] = mk.get("liquidityNum")
        book = _get_json(f"{CLOB_URL}/book?token_id={token}")
        asks = [(float(l["price"]), float(l["size"])) for l in book.get("asks", [])]
        bids = [(float(l["price"]), float(l["size"])) for l in book.get("bids", [])]
        montos = sorted({round(float(entry.get("bet_usd") or 0), 2), 50.0, 100.0})
        reg.update(analizar_libro(asks, bids, [m for m in montos if m > 0]))
    except Exception as e:
        reg["error"] = f"{type(e).__name__}: {e}"[:200]
    with open(BOOK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    if "error" in reg:
        print(f"  📘 Libro: no se pudo medir ({reg['error']})")
    else:
        print(f"  📘 Libro: ask={reg.get('best_ask')} spread={reg.get('spread')} "
              f"profundidad<=3c=${reg.get('depth_usd', {}).get('3c')}")


def run_one_cycle() -> dict:
    # run_cycle() de Bot 3 lee LOG_FILE y APUESTA_USD como globales del módulo
    # base en el momento de ejecutarse, así que basta con reasignarlos aquí.
    base.LOG_FILE = LOG_FILE_PROP
    capital = capital_simulado()
    base.APUESTA_USD = calcular_apuesta(capital)
    print(f"  💼 Bot 5 — capital simulado: ${capital:,.2f} → apuesta si hay señal: "
          f"${base.APUESTA_USD:.2f} ({STAKE_PCT*100:.0f}% con piso ${STAKE_MIN:.0f} / tope ${STAKE_MAX:.0f})")
    entry = base.run_cycle()
    if entry.get("action") == "BET":
        try:
            snapshot_libro(entry)
        except Exception as e:   # red de seguridad: medir nunca debe tumbar el ciclo
            print(f"  📘 Libro: fallo inesperado ({e})")
    return entry


def main():
    parser = argparse.ArgumentParser(description="BTC Scalp Proporcional (Bot 5)")
    parser.add_argument("--loop", action="store_true", help="Corre indefinidamente cada 5 minutos")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 BOT 5 — BTC SCALP PROPORCIONAL (paralelo a Bot 3)")
    print("=" * 60)
    print(f"  Misma señal que Bot 3; apuesta = {STAKE_PCT*100:.0f}% del capital "
          f"(piso ${STAKE_MIN:.0f}, tope ${STAKE_MAX:.0f})")
    print(f"  Log: {LOG_FILE_PROP}")

    if args.loop:
        try:
            while True:
                run_one_cycle()
                time.sleep(5 * 60)
        except KeyboardInterrupt:
            print("\n\n  🛑 Bot detenido.")
    else:
        result = run_one_cycle()
        if result.get("action") == "ERROR":
            sys.exit(1)


if __name__ == "__main__":
    main()
