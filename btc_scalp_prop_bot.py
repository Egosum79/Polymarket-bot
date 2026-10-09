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
from pathlib import Path

import btc_scalp_bot as base

STAKE_PCT        = 0.05    # fracción del capital simulado por apuesta
STAKE_MIN        = 10.0    # piso (igual al tamaño fijo de Bot 3)
STAKE_MAX        = 50.0    # tope provisional hasta medir liquidez real
CAPITAL_INICIAL  = 100.0   # mismo banco de arranque que el resto de los bots
BOT_TAG          = "bot5"  # etiqueta en settlements.jsonl (ver settle_bets.py)
LOG_FILE_PROP    = "btc_scalp_prop_log.jsonl"
SETTLEMENTS_FILE = "settlements.jsonl"


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


def run_one_cycle() -> dict:
    # run_cycle() de Bot 3 lee LOG_FILE y APUESTA_USD como globales del módulo
    # base en el momento de ejecutarse, así que basta con reasignarlos aquí.
    base.LOG_FILE = LOG_FILE_PROP
    capital = capital_simulado()
    base.APUESTA_USD = calcular_apuesta(capital)
    print(f"  💼 Bot 5 — capital simulado: ${capital:,.2f} → apuesta si hay señal: "
          f"${base.APUESTA_USD:.2f} ({STAKE_PCT*100:.0f}% con piso ${STAKE_MIN:.0f} / tope ${STAKE_MAX:.0f})")
    return base.run_cycle()


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
