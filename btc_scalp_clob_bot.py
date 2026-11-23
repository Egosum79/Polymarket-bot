#!/usr/bin/env python3
"""
=======================================================
  BOT 6 — BTC SCALP CON PRECIOS REALES DEL LIBRO (CLOB)
=======================================================

QUÉ ES:
  Misma señal que Bot 3 (modelo de 1 minuto, convicción mínima y veto de
  momentum, reutilizados de btc_scalp_bot.py), pero la ventaja y el precio
  de entrada se calculan contra el LIBRO DE ÓRDENES REAL (CLOB) en vez del
  precio de Gamma.

POR QUÉ EXISTE:
  Medición 2026-11-17/23 (btc_scalp_prop_book.jsonl, 66 apuestas de Bot 5):
  el ask real del CLOB estaba en promedio +6.0 centavos por encima del
  precio de Gamma con el que Bot 3/5 deciden y simulan (mediana +4.5c, hasta
  +22.5c). Repreciadas al ask real y con comisiones, esas 62 apuestas
  liquidadas pasaban de +$31.5 a -$233. Bot 3/5 podrían estar viendo "ventaja"
  que en realidad es un precio de Gamma atrasado. Este bot responde a la
  pregunta: ¿queda ventaja cuando se compra al precio que realmente se pagaría?

CÓMO DECIDE:
  Para cada lado (UP/DOWN) se simula comprar STAKE_USD recorriendo los asks
  del libro de ese lado (precio promedio de llenado, "vwap"). Costo efectivo
  por share = vwap + comisión taker, con comisión = FEE_RATE * vwap*(1-vwap)
  (fórmula de la documentación de Polymarket para cripto; NO verificada para
  estos mercados de 15 min -- el vwap crudo y la comisión por separado quedan
  en el log para poder recalcular con otra tarifa).
      edge_neto = P(nuestro lado gana) - costo_efectivo
  Se apuesta si edge_neto >= EDGE_NETO_MINIMO, hay convicción propia
  suficiente, y la apuesta no va en contra del momentum de la ventana (igual
  que Bot 3). Se elige el lado con mayor edge_neto; si el libro no alcanza a
  llenar la apuesta completa, ese lado no cuenta.

CÓMO SE LIQUIDA:
  settle_bets.py trata el log como los de Bot 3/5. Se guarda en market_prob el
  COSTO EFECTIVO por share (vwap + comisión): así el pnl liquidado
  (stake*(1/costo-1) si gana, -stake si pierde) ya es neto de comisión y de
  deslizamiento.

QUÉ SE REGISTRA (todos los ciclos, no solo las apuestas):
  Mejores bid/ask de UP y DOWN, precio de Gamma, probabilidad propia,
  momentum, etc. Eso permite re-simular offline otros umbrales de
  EDGE_NETO_MINIMO sin tener que esperar nuevas apuestas (solo hace falta el
  resultado de cada ventana, que se consulta en Gamma).

LÍMITES CONOCIDOS:
  - EDGE_NETO_MINIMO=0.05 es PROVISIONAL (no hay datos para calibrarlo).
  - Se asume ejecución inmediata al precio del libro del momento del ciclo;
    no modela latencia ni que el libro cambie mientras se envía la orden.
  - Apuesta fija de $10 a propósito: aquí se prueba la VENTAJA, no el tamaño.
  - Es simulación: no ejecuta órdenes reales.

USO:
  python btc_scalp_clob_bot.py            <- un ciclo
  python btc_scalp_clob_bot.py --loop     <- cada 5 minutos
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import btc_scalp_bot as base
from btc_scalp_prop_bot import _get_json, CLOB_URL

STAKE_USD         = 10.0     # apuesta fija (igual que Bot 3)
EDGE_NETO_MINIMO  = 0.05     # ventaja mínima DESPUÉS de deslizamiento y comisión (provisional)
FEE_RATE          = 0.07     # comisión taker: fee/share = FEE_RATE * p * (1-p)  (ver docstring)
LOG_FILE_CLOB     = "btc_scalp_clob_log.jsonl"
MOMENTUM_UMBRAL   = 0.01     # igual que Bot 3: |momentum de ventana| > 0.01% define dirección


# ─────────────────────────────────────────────────────
# LIBRO DE ÓRDENES
# ─────────────────────────────────────────────────────

def comprar_en_libro(asks: list[tuple[float, float]], usd: float) -> tuple[float | None, float]:
    """
    Simula comprar `usd` dólares recorriendo asks (precio, shares) de menor a
    mayor precio. Retorna (precio promedio de llenado, fracción llenada).
    """
    restante, gastado, shares = usd, 0.0, 0.0
    for precio, tam in sorted(asks):
        if restante <= 1e-9:
            break
        tomar = min(restante, precio * tam)
        gastado += tomar
        shares += tomar / precio
        restante -= tomar
    if shares <= 0:
        return None, 0.0
    return gastado / shares, gastado / usd


def costo_efectivo(vwap: float) -> tuple[float, float]:
    """(costo por share incluyendo comisión, comisión por share)."""
    fee = FEE_RATE * vwap * (1 - vwap)
    return vwap + fee, fee


def leer_libro(token: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    book = _get_json(f"{CLOB_URL}/book?token_id={token}")
    asks = [(float(l["price"]), float(l["size"])) for l in book.get("asks", [])]
    bids = [(float(l["price"]), float(l["size"])) for l in book.get("bids", [])]
    return asks, bids


def tokens_up_down(market: dict) -> tuple[str, str]:
    tokens = json.loads(market["clobTokenIds"])
    outcomes = [str(o).lower() for o in json.loads(market["outcomes"])]
    return tokens[outcomes.index("up")], tokens[outcomes.index("down")]


# ─────────────────────────────────────────────────────
# DECISIÓN (función pura, probada sin red)
# ─────────────────────────────────────────────────────

def evaluar(our_prob_up: float, window_momentum: float | None,
            asks_up: list, asks_down: list, stake: float = STAKE_USD) -> dict:
    """
    Decide con los precios reales del libro. Retorna un dict con la acción
    ('BET'/'PASS'), el lado elegido y todas las cifras que se registran.
    """
    lados = {}
    for lado, p_gana, asks in (("UP", our_prob_up, asks_up),
                               ("DOWN", 1 - our_prob_up, asks_down)):
        vwap, llenado = comprar_en_libro(asks, stake)
        if vwap is None or llenado < 0.999:
            lados[lado] = {"p": p_gana, "vwap": vwap, "llenado": round(llenado, 3), "ok": False}
            continue
        costo, fee = costo_efectivo(vwap)
        lados[lado] = {"p": p_gana, "vwap": vwap, "llenado": round(llenado, 3), "ok": True,
                       "costo": costo, "fee": fee, "edge_neto": p_gana - costo}

    elegibles = {k: v for k, v in lados.items() if v["ok"]}
    conviccion = abs(our_prob_up - 0.5)
    momentum_dir = None
    if window_momentum is not None:
        if window_momentum > MOMENTUM_UMBRAL:
            momentum_dir = "UP"
        elif window_momentum < -MOMENTUM_UMBRAL:
            momentum_dir = "DOWN"

    res = {"action": "PASS", "reason": None, "bet_side": None, "conviccion_propia": round(conviccion, 4),
           "contra_momentum": False, "lados": lados}
    if not elegibles:
        res["reason"] = "SIN_LIQUIDEZ"
        return res

    lado = max(elegibles, key=lambda k: elegibles[k]["edge_neto"])
    d = elegibles[lado]
    res["lado_mejor"] = lado
    res["contra_momentum"] = momentum_dir is not None and lado != momentum_dir

    if d["edge_neto"] < EDGE_NETO_MINIMO:
        res["reason"] = "EDGE_INSUFICIENTE"
    elif conviccion < base.CONVICCION_MINIMA:
        res["reason"] = "CONVICCION_INSUFICIENTE"
    elif res["contra_momentum"]:
        res["reason"] = "CONTRA_MOMENTUM"
    else:
        res["action"] = "BET"
        res["bet_side"] = lado
    return res


# ─────────────────────────────────────────────────────
# LOG
# ─────────────────────────────────────────────────────

def log_entry(entry: dict) -> None:
    with open(LOG_FILE_CLOB, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ya_aposto(market_id) -> bool:
    p = Path(LOG_FILE_CLOB)
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("market_id") == market_id and e.get("action") == "BET":
                return True
    return False


# ─────────────────────────────────────────────────────
# CICLO
# ─────────────────────────────────────────────────────

def run_cycle() -> dict:
    now = datetime.now(timezone.utc)
    print(f"\n{'─'*60}\n  🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}  (Bot 6, precios del CLOB)\n{'─'*60}")

    market = base.find_btc_window_market()
    if not market:
        print("  ⚠️  No hay ventana de 15 min activa ahora mismo")
        entry = {"timestamp": now.isoformat(), "action": "NO_MARKET"}
        log_entry(entry)
        return entry

    market_id = market.get("id")
    print(f"  🏪 {market.get('question','')[:70]}")

    if ya_aposto(market_id):
        print("  ⏭️  Ya hay una apuesta registrada sobre esta ventana — no se repite")
        entry = {"timestamp": now.isoformat(), "action": "SKIP", "reason": "YA_APOSTADO",
                 "market_id": market_id}
        log_entry(entry)
        return entry

    try:
        candles = base.get_1m_candles()
    except Exception as e:
        print(f"  ❌ Error descargando velas: {e}")
        entry = {"timestamp": now.isoformat(), "action": "ERROR", "reason": str(e)}
        log_entry(entry)
        return entry

    current_price = candles[-1]["close"]
    duration = base.parse_window_minutes(market.get("question", "")) or 15
    window_open_price = base.get_window_open_price(market, duration)
    analysis = base.analyze_scalp(candles, window_open_price, current_price)
    our_prob_up, used_model = base.our_probability_scalp(analysis)
    gamma_prob_up = base.get_market_probability(market, "UP")

    try:
        tok_up, tok_down = tokens_up_down(market)
        asks_up, bids_up = leer_libro(tok_up)
        asks_down, bids_down = leer_libro(tok_down)
    except Exception as e:
        print(f"  ❌ No se pudo leer el libro (CLOB): {e}")
        entry = {"timestamp": now.isoformat(), "action": "NO_LIBRO", "reason": f"{type(e).__name__}: {e}"[:200],
                 "market_id": market_id, "our_prob_up": our_prob_up, "gamma_prob_up": gamma_prob_up}
        log_entry(entry)
        return entry

    r = evaluar(our_prob_up, analysis["window_momentum"], asks_up, asks_down)
    lado = r.get("lado_mejor") or "UP"
    d = r["lados"][lado]
    print(f"  🧮 P(UP) propia {our_prob_up*100:.1f}% | Gamma UP {gamma_prob_up*100:.1f}¢ | "
          f"ask UP {min((p for p, _ in asks_up), default=0)*100:.0f}¢ / ask DOWN "
          f"{min((p for p, _ in asks_down), default=0)*100:.0f}¢")
    if d["ok"]:
        print(f"     Mejor lado {lado}: vwap {d['vwap']*100:.1f}¢ + comisión {d['fee']*100:.2f}¢ "
              f"-> edge neto {d['edge_neto']*100:+.1f}% | convicción {r['conviccion_propia']*100:.1f}%")
    print(f"  {'✅ APUESTA SIMULADA' if r['action']=='BET' else '⚪ ' + str(r['reason'])}"
          + (f": {r['bet_side']} ${STAKE_USD:.2f}" if r["action"] == "BET" else ""))

    bet = r["action"] == "BET"
    entry = {
        "timestamp":         now.isoformat(),
        "action":            r["action"],
        "reason":            r["reason"],
        "bet_side":          r["bet_side"],
        "bet_usd":           STAKE_USD if bet else 0,
        "our_prob_up":       our_prob_up,
        "our_prob":          d["p"],
        # costo efectivo por share (vwap + comisión): lo usa settle_bets.py como precio de entrada
        "market_prob":       round(d["costo"], 4) if d["ok"] else None,
        "fill_vwap":         round(d["vwap"], 4) if d["vwap"] else None,
        "fee_per_share":     round(d["fee"], 4) if d["ok"] else None,
        "edge_neto":         round(d["edge_neto"], 4) if d["ok"] else None,
        "lado_evaluado":     lado,
        "ask_up":            min((p for p, _ in asks_up), default=None),
        "ask_down":          min((p for p, _ in asks_down), default=None),
        "bid_up":            max((p for p, _ in bids_up), default=None),
        "bid_down":          max((p for p, _ in bids_down), default=None),
        "gamma_prob_up":     gamma_prob_up,
        "conviccion_propia": r["conviccion_propia"],
        "contra_momentum":   r["contra_momentum"],
        "btc_price":         current_price,
        "rsi":               analysis["rsi"],
        "ema_diff":          analysis["ema_diff"],
        "window_momentum":   analysis["window_momentum"],
        "market_id":         market_id,
        "market_q":          market.get("question", "")[:80],
        "market_end":        market.get("endDate"),
        "used_model":        used_model,
    }
    log_entry(entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description="BTC Scalp con precios del CLOB (Bot 6)")
    parser.add_argument("--loop", action="store_true", help="Corre indefinidamente cada 5 minutos")
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 BOT 6 — BTC SCALP CON PRECIOS REALES DEL LIBRO")
    print("=" * 60)
    print(f"  Apuesta ${STAKE_USD:.0f} | edge neto mínimo {EDGE_NETO_MINIMO*100:.0f}% | comisión {FEE_RATE}*p*(1-p)")
    print(f"  Log: {LOG_FILE_CLOB}")

    if args.loop:
        try:
            while True:
                run_cycle()
                time.sleep(5 * 60)
        except KeyboardInterrupt:
            print("\n\n  🛑 Bot detenido.")
    else:
        result = run_cycle()
        if result.get("action") == "ERROR":
            sys.exit(1)


if __name__ == "__main__":
    main()
