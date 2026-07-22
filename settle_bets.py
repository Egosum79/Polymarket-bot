#!/usr/bin/env python3
"""
=======================================================
  LIQUIDADOR DE APUESTAS SIMULADAS
  Revisa mercados de Polymarket ya resueltos y calcula el
  P&L real (ganancia/pérdida en USD) de las apuestas que
  polymarket_bot.py y btc_direction_bot.py registraron.
=======================================================

Por qué existe:
  bot_log.jsonl / btc_bot_log.jsonl solo registran "qué apuesta
  habríamos hecho" en el momento de la señal. Nunca vuelven a
  revisar si esa apuesta ganó o perdió. Este script cierra ese
  ciclo: para cada apuesta con un market_id real, consulta si el
  mercado ya se resolvió en Polymarket y, si es así, calcula su
  resultado y lo guarda en settlements.jsonl (cache incremental,
  para no volver a consultar mercados ya liquidados).

  daily_review.py lee settlements.jsonl para mostrar P&L
  acumulado, ROI% y tasa de acierto en el reporte diario, además
  de una curva de capital simulado (arranca en $100 por bot y va
  sumando/restando el pnl de cada apuesta liquidada, en el orden
  en que se resolvió — campo 'settled_at').

LIMITACIÓN CONOCIDA:
  Cada ciclo del bot registra su señal como una apuesta
  independiente, incluso si ya existe una señal previa sobre el
  mismo mercado (no hay gestión de posiciones abiertas). El P&L
  aquí calculado asume que CADA señal se habría apostado por
  separado — sobreestima el capital desplegado si en la práctica
  uno solo tomaría la posición una vez por mercado. La curva de
  capital hereda la misma limitación: no reserva capital para
  apuestas todavía abiertas, así que puede mostrar más exposición
  simultánea de la que $100 reales permitirían.

USO:
  python settle_bets.py
"""

import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# En consolas Windows con codepage legado (cp1252), imprimir emojis revienta
# con UnicodeEncodeError. Forzamos stdout/stderr a UTF-8 si el terminal lo permite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GAMMA_URL         = "https://gamma-api.polymarket.com"
SETTLEMENTS_FILE  = "settlements.jsonl"
BOT1_LOG          = "bot_log.jsonl"
BOT2_LOG          = "btc_bot_log.jsonl"


def fetch(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
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


def bet_key(bot: str, entry: dict) -> str:
    """Identificador único de una apuesta simulada (bot + mercado + momento de la señal)."""
    return f"{bot}:{entry.get('market_id')}:{entry.get('timestamp')}"


def get_market(market_id) -> dict | None:
    try:
        return fetch(f"{GAMMA_URL}/markets/{market_id}")
    except Exception as e:
        print(f"  ⚠️  Error consultando mercado {market_id}: {e}")
        return None


def resolve_outcome(market: dict) -> str | None:
    """Retorna 'YES' o 'NO' según el lado ganador, o None si el mercado no está resuelto."""
    if not market.get("closed"):
        return None
    try:
        prices    = json.loads(market.get("outcomePrices", "[]"))
        yes_price = float(prices[0])
        no_price  = float(prices[1])
    except Exception:
        return None
    return "YES" if yes_price > no_price else "NO"


def pnl_for_bet(bet_side_won: bool, bet_price: float, bet_usd: float) -> float:
    """P&L en USD de una apuesta binaria de Polymarket: se compran shares a bet_price
    (cada share vale $1 si gana, $0 si pierde)."""
    if not bet_price or bet_price <= 0:
        return 0.0
    if bet_side_won:
        return bet_usd * (1.0 / bet_price - 1.0)
    return -bet_usd


def settle_bot1(entries: list[dict], already_settled: set) -> list[dict]:
    """bot_log.jsonl: bet_side ya es YES/NO directamente sobre el mercado."""
    results = []
    for e in entries:
        market_id = e.get("market_id")
        if not market_id:
            continue
        key = bet_key("bot1", e)
        if key in already_settled:
            continue
        market = get_market(market_id)
        if market is None:
            continue
        outcome = resolve_outcome(market)
        if outcome is None:
            continue   # aún no resuelto, se reintenta en el próximo ciclo
        won = (e.get("bet_side") == outcome)
        pnl = pnl_for_bet(won, e.get("bet_price", 0), e.get("bet_usd", 0))
        results.append({
            "key":        key,
            "bot":        "bot1",
            "market_id":  market_id,
            "question":   e.get("question", ""),
            "bet_side":   e.get("bet_side"),
            "bet_usd":    e.get("bet_usd", 0),
            "outcome":    outcome,
            "won":        won,
            "pnl":        round(pnl, 4),
            "timestamp":  e.get("timestamp"),
            "settled_at": datetime.now(timezone.utc).isoformat(),
        })
    return results


def settle_bot2(entries: list[dict], already_settled: set) -> list[dict]:
    """btc_bot_log.jsonl: bet_side es UP/DOWN (UP↔YES, DOWN↔NO en el mercado 'Up or Down')."""
    results = []
    for e in entries:
        if e.get("action") != "BET":
            continue
        market_id = e.get("market_id")
        if not market_id:
            continue   # sin mercado real emparejado, no se puede liquidar contra Polymarket
        key = bet_key("bot2", e)
        if key in already_settled:
            continue
        market = get_market(market_id)
        if market is None:
            continue
        outcome = resolve_outcome(market)
        if outcome is None:
            continue
        bet_side_yn = "YES" if e.get("bet_side") == "UP" else "NO"
        won = (bet_side_yn == outcome)
        pnl = pnl_for_bet(won, e.get("market_prob", 0), e.get("bet_usd", 0))
        results.append({
            "key":        key,
            "bot":        "bot2",
            "market_id":  market_id,
            "question":   e.get("market_q", ""),
            "bet_side":   e.get("bet_side"),
            "bet_usd":    e.get("bet_usd", 0),
            "outcome":    "UP" if outcome == "YES" else "DOWN",
            "won":        won,
            "pnl":        round(pnl, 4),
            "timestamp":  e.get("timestamp"),
            "settled_at": datetime.now(timezone.utc).isoformat(),
        })
    return results


def main():
    print("=" * 60)
    print("  LIQUIDADOR DE APUESTAS SIMULADAS")
    print("=" * 60)

    settled_so_far  = load_jsonl(SETTLEMENTS_FILE)
    already_settled = {s["key"] for s in settled_so_far}

    bot1_entries = load_jsonl(BOT1_LOG)
    bot2_entries = load_jsonl(BOT2_LOG)

    new_results = settle_bot1(bot1_entries, already_settled) + settle_bot2(bot2_entries, already_settled)

    print(f"  Apuestas liquidadas previamente: {len(settled_so_far)}")
    print(f"  Mercados aún pendientes de resolver o recién liquidados: revisando...")
    print(f"  Apuestas recién resueltas en este ciclo: {len(new_results)}")

    if new_results:
        with open(SETTLEMENTS_FILE, "a", encoding="utf-8") as f:
            for r in new_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in new_results:
            icon = "✅" if r["won"] else "❌"
            print(f"  {icon} [{r['bot']}] {r['bet_side']} → resultado {r['outcome']} "
                  f"| P&L: ${r['pnl']:+.2f} | {r['question'][:60]}")

    print("\n  ✅ Liquidación completada.")


if __name__ == "__main__":
    main()
