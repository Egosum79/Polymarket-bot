#!/usr/bin/env python3
"""
=======================================================
  POLYMARKET BITCOIN BOT
  Bot automático para mercados de precio de Bitcoin
=======================================================

MODOS:
  - SIMULACIÓN (por defecto): muestra qué haría sin gastar dinero
  - REAL: ejecuta apuestas reales via Polymarket CLOB API

ESTRATEGIA:
  Compara el precio real de Bitcoin con la probabilidad
  implícita en los mercados de Polymarket. Si el mercado
  cree que BTC llegará a $X pero el precio actual y la
  tendencia sugieren lo contrario → apuesta en contra.

INSTALACIÓN (solo para modo REAL):
  pip install py-clob-client --break-system-packages

USO:
  python polymarket_bot.py              ← modo simulación
  python polymarket_bot.py --real       ← modo real (requiere API keys)
  python polymarket_bot.py --intervalo 60  ← cada 60 minutos
"""

import json
import sys
import time
import math
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# En consolas Windows con codepage legado (cp1252), imprimir emojis revienta
# con UnicodeEncodeError. Forzamos stdout/stderr a UTF-8 si el terminal lo permite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

# ── Polymarket API ─────────────────────────────────
GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL  = "https://clob.polymarket.com"

# ── Credenciales (solo modo REAL) ──────────────────
# Obtén tu API key en: polymarket.com → Perfil → API
# NUNCA compartas estas credenciales con nadie
API_KEY      = "TU_API_KEY_AQUI"
API_SECRET   = "TU_API_SECRET_AQUI"
API_PASSPHRASE = "TU_PASSPHRASE_AQUI"
PRIVATE_KEY  = "TU_PRIVATE_KEY_WALLET"   # clave privada de tu wallet Polygon

# ── Parámetros del bot ─────────────────────────────
APUESTA_MIN_USD    = 5.0     # mínimo a apostar por señal
APUESTA_MAX_USD    = 20.0    # máximo por señal
CAPITAL_TOTAL      = 100.0   # capital disponible
MAX_POSICIONES     = 5       # máximo de posiciones abiertas simultáneas
EDGE_MINIMO        = 0.08    # ventaja mínima para apostar (8 puntos porcentuales)
INTERVALO_MINUTOS  = 30      # cada cuántos minutos revisar

# ── Palabras clave para filtrar mercados de BTC ───
BTC_KEYWORDS = [
    "bitcoin", "btc", "will bitcoin",
    "bitcoin reach", "bitcoin dip", "bitcoin hit"
]

LOG_FILE = "bot_log.jsonl"   # log de todas las decisiones


# ─────────────────────────────────────────────────────
# PRECIO REAL DE BITCOIN
# ─────────────────────────────────────────────────────

def get_btc_price() -> dict:
    """
    Obtiene precio actual de BTC desde CoinGecko (gratis, sin API key).
    Retorna: precio USD, cambio 24h, cambio 7d
    """
    url = ("https://api.coingecko.com/api/v3/simple/price"
           "?ids=bitcoin&vs_currencies=usd"
           "&include_24hr_change=true&include_7d_change=true")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())["bitcoin"]
            return {
                "price":     data["usd"],
                "change_24h": data.get("usd_24h_change", 0),
                "change_7d":  data.get("usd_7d_change", 0),
            }
    except Exception as e:
        print(f"  ⚠️  Error obteniendo precio BTC: {e}")
        return {"price": None, "change_24h": 0, "change_7d": 0}


# ─────────────────────────────────────────────────────
# MERCADOS DE BITCOIN EN POLYMARKET
# ─────────────────────────────────────────────────────

def fetch(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_btc_history(days: int = 30) -> list:
    """Descarga histórico de precios BTC (CoinGecko market_chart). Retorna lista [timestamp_ms, price]."""
    url = f"{'https://api.coingecko.com/api/v3'}/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
    try:
        data = fetch(url)
        return data.get("prices", [])
    except Exception as e:
        print(f"  ⚠️  Error obteniendo histórico BTC: {e}")
        return []


def estimate_btc_drift_vol(prices: list) -> tuple[float, float]:
    """
    Estima drift (mu) y volatilidad (sigma) DIARIOS de BTC a partir de precios
    históricos, usando retornos logarítmicos entre puntos consecutivos (la
    frecuencia real se mide desde los timestamps, sea horaria o diaria).

    El drift se encoge (shrinkage) hacia 0: el retorno medio de una muestra
    corta es un estimador muy ruidoso de la tendencia real (su error estándar
    es del mismo orden que la señal misma). La volatilidad es un estimador
    mucho más estable y es la que hace el trabajo pesado del modelo.
    """
    if len(prices) < 10:
        return 0.0, 0.02   # fallback conservador (~2% diario, típico de BTC)

    closes = [p[1] for p in prices]
    times  = [p[0] / 86_400_000 for p in prices]   # ms → días

    log_returns, dts = [], []
    for i in range(1, len(closes)):
        dt = times[i] - times[i - 1]
        if dt <= 0 or closes[i - 1] <= 0:
            continue
        log_returns.append(math.log(closes[i] / closes[i - 1]))
        dts.append(dt)

    if not log_returns:
        return 0.0, 0.02

    avg_dt = sum(dts) / len(dts)
    mean_r = sum(log_returns) / len(log_returns)
    var_r  = sum((r - mean_r) ** 2 for r in log_returns) / max(1, len(log_returns) - 1)

    mu_daily    = (mean_r / avg_dt) * 0.25   # shrinkage — ver docstring
    sigma_daily = math.sqrt(var_r / avg_dt)

    return mu_daily, max(sigma_daily, 0.005)


def norm_cdf(x: float) -> float:
    """CDF de la normal estándar, vía la función error (sin dependencias externas)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def get_btc_markets() -> list[dict]:
    """Descarga mercados activos de Bitcoin en Polymarket."""
    url = f"{GAMMA_URL}/markets?limit=100&active=true&order=volume24hr&ascending=false"
    try:
        markets = fetch(url)
    except Exception as e:
        print(f"  ❌ Error descargando mercados: {e}")
        return []

    btc_markets = []
    for m in markets:
        question = m.get("question", "").lower()
        if any(kw in question for kw in BTC_KEYWORDS):
            btc_markets.append(m)

    return btc_markets


# ─────────────────────────────────────────────────────
# ANÁLISIS DE OPORTUNIDAD
# ─────────────────────────────────────────────────────

def parse_target_price(question: str) -> float | None:
    """
    Extrae el precio objetivo del título del mercado.
    Ej: "Will Bitcoin reach $150,000?" → 150000
    """
    import re
    # Busca patrones como $150,000 o $150k
    patterns = [
        r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)',   # $150,000
        r'\$([0-9]+)k\b',                                  # $150k
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                price = float(val)
                if "k" in pat:
                    price *= 1000
                return price
            except ValueError:
                continue
    return None


def detect_direction(question: str) -> str:
    """
    Detecta si el mercado es de subida o bajada.
    Retorna: 'UP', 'DOWN', o 'UNKNOWN'
    """
    q = question.lower()
    if any(w in q for w in ["reach", "hit", "exceed", "above", "high", "surpass"]):
        return "UP"
    if any(w in q for w in ["dip", "drop", "fall", "below", "low", "crash"]):
        return "DOWN"
    return "UNKNOWN"


def analyze_btc_market(market: dict, btc: dict, mu: float, sigma: float) -> dict | None:
    """
    Analiza si hay una oportunidad en un mercado de BTC.

    Modelo: BTC sigue un movimiento geométrico browniano (GBM) con drift `mu`
    y volatilidad `sigma` diarios, estimados de precios históricos reales
    (ver estimate_btc_drift_vol). Bajo GBM, ln(S_T) ~ Normal(ln(S0) + mu*T,
    sigma²*T), lo que da una probabilidad de llegar al precio objetivo mucho
    mejor fundamentada que una heurística de "tendencia reciente" arbitraria.

    El "edge" es la diferencia entre la probabilidad implícita del mercado
    y nuestra estimación de la probabilidad real (nuestra_prob).
    """
    if btc["price"] is None:
        return None

    question = market.get("question", "")
    try:
        prices     = json.loads(market.get("outcomePrices", "[0.5,0.5]"))
        yes_price  = float(prices[0])
        no_price   = float(prices[1])
    except Exception:
        return None

    # Filtros básicos
    if yes_price < 0.04 or yes_price > 0.96:
        return None   # ya resuelto
    liquidity = market.get("liquidityNum", 0) or 0
    if liquidity < 5000:
        return None
    spread = market.get("spread", 99) or 99
    if spread > 0.06:
        return None

    # Días restantes
    end_date = market.get("endDate", "")
    try:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        days_left = max(0, (end - datetime.now(timezone.utc)).total_seconds() / 86400)
    except Exception:
        days_left = 0
    if days_left < 1:
        return None

    target_price = parse_target_price(question)
    direction    = detect_direction(question)

    if target_price is None or direction == "UNKNOWN":
        return None

    btc_price    = btc["price"]
    change_24h   = btc["change_24h"] / 100   # como decimal, solo informativo
    change_7d    = btc["change_7d"] / 100

    # ── Estimación de probabilidad real (GBM log-normal) ──────────────
    T = max(days_left, 0.25)   # evita división por T≈0 en mercados que vencen ya
    d = (math.log(btc_price / target_price) + mu * T) / (sigma * math.sqrt(T))
    prob_above_target = norm_cdf(d)   # P(BTC_T >= target)

    if direction == "UP":
        our_prob = prob_above_target
    elif direction == "DOWN":
        our_prob = 1 - prob_above_target
    else:
        return None

    our_prob = max(0.02, min(0.98, our_prob))

    if our_prob < yes_price - EDGE_MINIMO:
        # El mercado sobrevalora la probabilidad → BET NO
        edge = yes_price - our_prob
        bet_side = "NO"
        bet_price = no_price
    elif our_prob > yes_price + EDGE_MINIMO:
        # El mercado subvalora → BET YES
        edge = our_prob - yes_price
        bet_side = "YES"
        bet_price = yes_price
    else:
        return None   # sin ventaja suficiente

    # ── Tamaño de apuesta (Kelly simplificado) ────────
    # f = (edge * (1/bet_price - 1) - (1-edge)) / (1/bet_price - 1)
    # Limitado a APUESTA_MIN/MAX
    odds = 1.0 / bet_price
    kelly = (our_prob * (odds - 1) - (1 - our_prob)) / (odds - 1)
    kelly = max(0, kelly)
    bet_usd = min(APUESTA_MAX_USD, max(APUESTA_MIN_USD, CAPITAL_TOTAL * kelly * 0.25))

    return {
        "market_id":     market.get("id"),
        "question":      question,
        "slug":          market.get("slug", ""),
        "direction":     direction,
        "target_price":  target_price,
        "btc_price":     btc_price,
        "yes_price":     round(yes_price, 4),
        "no_price":      round(no_price, 4),
        "our_prob":      round(our_prob, 4),
        "edge":          round(edge, 4),
        "bet_side":      bet_side,
        "bet_price":     round(bet_price, 4),
        "bet_usd":       round(bet_usd, 2),
        "days_left":     round(days_left, 1),
        "liquidity":     liquidity,
        "change_24h":    round(change_24h * 100, 2),
        "change_7d":     round(change_7d * 100, 2),
    }


# ─────────────────────────────────────────────────────
# EJECUCIÓN REAL (CLOB API)
# ─────────────────────────────────────────────────────

def place_order_real(signal: dict) -> bool:
    """
    Coloca una orden real en Polymarket via CLOB API.
    Requiere: pip install py-clob-client
    """
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.constants import POLYGON

        client = ClobClient(
            host       = CLOB_URL,
            chain_id   = POLYGON,
            key        = PRIVATE_KEY,
            signature_type = 1,   # POLY_PROXY
            funder     = None,
        )
        client.set_api_creds(client.create_or_derive_api_creds())

        # Obtener token ID del lado correcto
        market_data = fetch(f"{GAMMA_URL}/markets/{signal['market_id']}")
        token_ids   = json.loads(market_data.get("clobTokenIds", "[]"))
        token_id    = token_ids[0] if signal["bet_side"] == "YES" else token_ids[1]

        order = client.create_and_post_order(OrderArgs(
            token_id = token_id,
            price    = signal["bet_price"],
            size     = signal["bet_usd"] / signal["bet_price"],
            side     = "BUY",
            order_type = OrderType.FOK,   # Fill or Kill
        ))
        print(f"  ✅ ORDEN EJECUTADA: {signal['bet_side']} ${signal['bet_usd']:.2f}")
        print(f"     Order ID: {order.get('orderID', 'N/A')}")
        return True

    except ImportError:
        print("  ❌ Falta instalar: pip install py-clob-client --break-system-packages")
        return False
    except Exception as e:
        print(f"  ❌ Error colocando orden: {e}")
        return False


# ─────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────

def log_signal(signal: dict, action: str, mode: str):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode":      mode,
        "action":    action,
        **signal
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────
# CICLO PRINCIPAL DEL BOT
# ─────────────────────────────────────────────────────

def run_cycle(mode: str = "simulation") -> list[dict]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'─'*60}")
    print(f"  🤖 CICLO BOT  [{now}]  MODO: {mode.upper()}")
    print(f"{'─'*60}")

    # 1. Precio actual de Bitcoin
    btc = get_btc_price()
    if btc["price"]:
        print(f"\n  💰 Bitcoin: ${btc['price']:,.0f}  "
              f"({btc['change_24h']:+.1f}% 24h / {btc['change_7d']:+.1f}% 7d)")
    else:
        print("  ❌ No se pudo obtener precio de BTC")
        return []

    # 2. Volatilidad y drift reales (histórico 30 días, una sola vez por ciclo)
    history     = get_btc_history(30)
    mu, sigma   = estimate_btc_drift_vol(history)
    print(f"  📐 Modelo GBM: drift diario {mu*100:+.3f}%  |  volatilidad diaria {sigma*100:.2f}%")

    # 3. Mercados de BTC en Polymarket
    markets = get_btc_markets()
    print(f"  📊 Mercados BTC encontrados: {len(markets)}")

    # 4. Analizar cada mercado
    signals = []
    for m in markets:
        result = analyze_btc_market(m, btc, mu, sigma)
        if result:
            signals.append(result)

    if not signals:
        print("\n  ⚪ Sin oportunidades en este ciclo (edge < {:.0f}%)".format(
            EDGE_MINIMO * 100))
        return []

    print(f"\n  🎯 {len(signals)} oportunidad(es) detectada(s):\n")

    executed = []
    for s in signals:
        link = f"https://polymarket.com/event/{s['slug']}"
        print(f"  {'🟢' if s['bet_side']=='YES' else '🔴'} BET {s['bet_side']}  |  Edge: {s['edge']*100:.1f}%  |  ${s['bet_usd']:.2f}")
        print(f"     {s['question'][:70]}")
        print(f"     BTC: ${s['btc_price']:,.0f} → Target: ${s['target_price']:,.0f}  ({s['direction']})")
        print(f"     Mercado: {int(s['yes_price']*100)}¢ YES  |  Nuestra estima: {int(s['our_prob']*100)}¢")
        print(f"     Liquidez: ${s['liquidity']:,.0f}  |  Días: {s['days_left']:.0f}  |  {link}")

        if mode == "simulation":
            print(f"     [SIMULACIÓN] Apostaría ${s['bet_usd']:.2f} en {s['bet_side']}")
            log_signal(s, "SIMULATED", mode)
            executed.append(s)
        else:
            print(f"     Ejecutando orden real...")
            ok = place_order_real(s)
            if ok:
                log_signal(s, "EXECUTED", mode)
                executed.append(s)
            else:
                log_signal(s, "FAILED", mode)
        print()

    return executed


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket Bitcoin Bot")
    parser.add_argument("--real",       action="store_true",
                        help="Modo real (ejecuta apuestas reales)")
    parser.add_argument("--intervalo",  type=int, default=INTERVALO_MINUTOS,
                        help="Minutos entre ciclos (default: 30)")
    parser.add_argument("--ciclos",     type=int, default=0,
                        help="Número de ciclos (0 = infinito)")
    args = parser.parse_args()

    mode     = "real" if args.real else "simulation"
    intervalo = args.intervalo
    max_ciclos = args.ciclos

    print("=" * 60)
    print("  🤖 POLYMARKET BITCOIN BOT")
    print("=" * 60)
    print(f"  Modo:      {mode.upper()}")
    print(f"  Intervalo: {intervalo} minutos")
    print(f"  Edge mín:  {EDGE_MINIMO*100:.0f}%")
    print(f"  Apuesta:   ${APUESTA_MIN_USD:.0f} – ${APUESTA_MAX_USD:.0f} por señal")
    print(f"  Capital:   ${CAPITAL_TOTAL:.0f}")
    if mode == "real":
        print("\n  ⚠️  MODO REAL ACTIVADO — Se ejecutarán apuestas reales")
        print("  Presiona Ctrl+C para cancelar en los próximos 5 segundos...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  ❌ Cancelado.")
            return
    print(f"\n  Log guardado en: {LOG_FILE}")
    print("  Presiona Ctrl+C para detener el bot\n")

    ciclo = 0
    try:
        while True:
            ciclo += 1
            run_cycle(mode)

            if max_ciclos > 0 and ciclo >= max_ciclos:
                print(f"\n  ✅ {ciclo} ciclo(s) completados. Bot detenido.")
                break

            print(f"\n  ⏰ Próximo ciclo en {intervalo} minutos... (Ctrl+C para detener)")
            time.sleep(intervalo * 60)

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Bot detenido manualmente tras {ciclo} ciclo(s).")
        print(f"  Log completo en: {LOG_FILE}")


if __name__ == "__main__":
    main()
