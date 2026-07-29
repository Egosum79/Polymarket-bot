#!/usr/bin/env python3
"""
esports_bot.py - Bot de predicciones de esports para Polymarket.

Descubre mercados de esports en Polymarket (Gamma API), calcula una
probabilidad propia usando estadisticas de PandaScore, compara contra el
precio de mercado y decide si apostar. Funciona en modo simulacion por
defecto; con --real intenta ejecutar la orden vía py-clob-client.

Solo libreria estandar (urllib, json, re, datetime, argparse) para poder
correr en GitHub Actions sin pasos de instalacion adicionales.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------

GAMMA_MARKETS_URL = (
    "https://gamma-api.polymarket.com/markets"
    "?limit=100&active=true&closed=false&order=volume24hr&ascending=false"
)
GAMMA_PAGE_SIZE = 100  # la API ignora limit>100 y siempre trunca a 100 por pagina
GAMMA_MAX_PAGES = 5    # hasta 500 mercados; suficiente para no perder esports de bajo volumen
PANDASCORE_MATCHES_URL = "https://api.pandascore.co/matches"   # OJO: es .co, no .io (ver diagnostico 2026-07-28)

LOG_FILE = "esports_bot_log.jsonl"

EDGE_MINIMO = 0.07
BET_USD = 8.0
MIN_LIQUIDITY = 5000.0
MIN_PRICE = 0.10
MAX_PRICE = 0.90
HORAS_MAX_RESOLUCION = 24  # antes 6: excluia LCK/LPL, que abren mercado ~10-14h antes

REQUEST_TIMEOUT = 15

# Palabras que indican que el mercado es un enfrentamiento (partido) concreto
KEYWORDS_MATCH = ["vs", "bo1", "bo3", "bo5", "match", "game"]

# Palabras/tags que indican que el mercado es de esports, mapeadas al nombre
# del juego para el log. Se revisan en orden, la primera coincidencia gana.
KEYWORDS_GAME = [
    (["league of legends", "lol", "lck", "lpl", "lcs"], "League of Legends"),
    (["dota"], "Dota 2"),
    (["cs2", "counter-strike", "csgo", "cs:go"], "CS2"),
    (["valorant", "vct"], "Valorant"),
    (["kespa"], "KeSPA"),
    (["epl"], "EPL"),
    (["esports"], "Esports"),
]

# Equipos conocidos y su region "casa", usado como proxy de ventaja regional.
# Heuristica best-effort: si el nombre de la liga/torneo del partido menciona
# esa region, el equipo recibe puntaje de "home advantage".
EQUIPO_REGION = {
    "t1": "korea", "gen.g": "korea", "gen g": "korea", "drx": "korea",
    "kt rolster": "korea", "dplus koz": "korea", "hanwha life esports": "korea",
    "kwangdong freecs": "korea", "dn freecs": "korea",
    "g2 esports": "europe", "g2": "europe", "fnatic": "europe",
    "mad lions": "europe", "team vitality": "europe", "vitality": "europe",
    "top esports": "china", "tes": "china", "jdg": "china",
    "jd gaming": "china", "bilibili gaming": "china", "lng esports": "china",
    "cloud9": "north america", "c9": "north america",
    "team liquid": "north america", "tl": "north america",
    "100 thieves": "north america", "evil geniuses": "north america",
}

REGION_KEYWORDS = {
    "korea": ["lck", "korea", "korean"],
    "china": ["lpl", "china", "chinese"],
    "europe": ["lec", "europe", "european"],
    "north america": ["lcs", "north america", "na "],
}


# ----------------------------------------------------------------------
# Utilidades HTTP
# ----------------------------------------------------------------------

def fetch_json(url, headers=None):
    """Hace un GET y parsea JSON. Devuelve None si algo falla (nunca lanza)."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "esports_bot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        return json.loads(data)
    except urllib.error.HTTPError as e:
        print(f"  [HTTP {e.code}] {url}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"  [URLError] {url}: {e.reason}", file=sys.stderr)
        return None
    except (TimeoutError, json.JSONDecodeError, Exception) as e:
        print(f"  [Error] {url}: {e}", file=sys.stderr)
        return None


# ----------------------------------------------------------------------
# Descubrimiento de mercados
# ----------------------------------------------------------------------

def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_list_field(raw):
    """Gamma API a veces devuelve listas como string JSON; normaliza a lista."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _tag_text(tags):
    partes = []
    for t in tags:
        if isinstance(t, dict):
            partes.append(str(t.get("label", "")))
            partes.append(str(t.get("slug", "")))
        else:
            partes.append(str(t))
    return " ".join(partes).lower()


def es_mercado_esports(question_lower, tags_text):
    tiene_match = any(k in question_lower for k in KEYWORDS_MATCH)
    juego_texto = question_lower + " " + tags_text
    tiene_juego = any(
        any(k in juego_texto for k in kws) for kws, _ in KEYWORDS_GAME
    )
    return tiene_match and tiene_juego


def detectar_juego(question_lower, tags_text):
    texto = question_lower + " " + tags_text
    for kws, nombre in KEYWORDS_GAME:
        if any(k in texto for k in kws):
            return nombre
    return "Esports"


def fetch_todos_los_mercados():
    """
    Descarga mercados activos de Gamma API paginando con offset.

    La API ignora silenciosamente limit>100 y siempre trunca a 100 items
    por respuesta (confirmado 2026-07-29: pedir limit=200 o limit=500 igual
    devuelve 100). Sin paginacion, cualquier mercado de esports fuera de los
    100 de mayor volumen 24h -- asi sea de una liga de primer nivel -- nunca
    se llega a evaluar. Pagina hasta GAMMA_MAX_PAGES paginas o hasta que la
    API devuelva menos de una pagina completa (fin de resultados).
    """
    todos = []
    for pagina in range(GAMMA_MAX_PAGES):
        offset = pagina * GAMMA_PAGE_SIZE
        url = f"{GAMMA_MARKETS_URL}&offset={offset}"
        data = fetch_json(url)
        if not data or not isinstance(data, list):
            break
        todos.extend(data)
        if len(data) < GAMMA_PAGE_SIZE:
            break
    return todos


def fetch_esports_markets():
    """Descarga mercados activos de Gamma API y filtra los de esports validos."""
    data = fetch_todos_los_mercados()
    if not data:
        return []

    ahora = datetime.now(timezone.utc)
    limite_resolucion = ahora + timedelta(hours=HORAS_MAX_RESOLUCION)

    resultado = []
    for m in data:
        try:
            question = m.get("question") or ""
            question_lower = question.lower()
            tags = _load_list_field(m.get("tags"))
            tags_text = _tag_text(tags)

            if not es_mercado_esports(question_lower, tags_text):
                continue

            end_date_raw = m.get("endDate")
            if not end_date_raw:
                continue
            try:
                end_date = datetime.fromisoformat(end_date_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            # Excluir mercados "zombie": endDate ya paso pero Polymarket
            # todavia no lo marco closed=true (visto 2026-07-29 con
            # "Map Winner" de CS2 con horas_restantes negativas).
            if end_date <= ahora:
                continue
            if end_date > limite_resolucion:
                continue

            outcomes = _load_list_field(m.get("outcomes"))
            outcome_prices = _load_list_field(m.get("outcomePrices"))
            if not outcome_prices:
                continue
            yes_price = _to_float(outcome_prices[0], -1)
            if yes_price < 0:
                continue
            # Excluir mercados casi decididos (99c / 1c) y fuera del rango de interes
            if yes_price < MIN_PRICE or yes_price > MAX_PRICE:
                continue

            liquidity = _to_float(m.get("liquidity", m.get("liquidityNum")))
            if liquidity < MIN_LIQUIDITY:
                continue

            volume24h = _to_float(m.get("volume24hr", m.get("volume24hrClob")))

            token_ids = _load_list_field(m.get("clobTokenIds"))

            resultado.append({
                "id": m.get("id"),
                "slug": m.get("slug"),
                "question": question,
                "juego": detectar_juego(question_lower, tags_text),
                "outcomes": outcomes,
                "yes_price": yes_price,
                "liquidity": liquidity,
                "volume24h": volume24h,
                "end_date": end_date_raw,
                "token_ids": token_ids,
            })
        except Exception as e:
            print(f"  [Error procesando mercado] {e}", file=sys.stderr)
            continue

    return resultado


# ----------------------------------------------------------------------
# Extraccion de equipos desde la pregunta
# ----------------------------------------------------------------------

_SUFIJOS_RUIDO = re.compile(
    r"\s*[-–]\s*(game|map|bo1|bo3|bo5)\b.*$|\s*\(.*?\)\s*$|\?\s*$",
    re.IGNORECASE,
)


def _limpiar_nombre_equipo(nombre):
    nombre = _SUFIJOS_RUIDO.sub("", nombre)
    return nombre.strip(" .,-")


def extraer_equipos(question):
    """Extrae 'Team A' y 'Team B' de una pregunta tipo 'Team A vs Team B'."""
    match = re.search(r"(.+?)\s+vs\.?\s+(.+)", question, re.IGNORECASE)
    if not match:
        return None
    equipo_a = _limpiar_nombre_equipo(match.group(1))
    equipo_b = _limpiar_nombre_equipo(match.group(2))
    if not equipo_a or not equipo_b:
        return None
    return equipo_a, equipo_b


# ----------------------------------------------------------------------
# Estadisticas via PandaScore
# ----------------------------------------------------------------------

PANDASCORE_TEAMS_URL = "https://api.pandascore.co/teams"


# Mapeo del nombre de juego que usa el bot (ver KEYWORDS_GAME) al slug de
# videojuego que usa PandaScore en sus endpoints por juego (/lol/teams,
# /csgo/teams, etc). None = sin endpoint por juego conocido para probar.
PANDASCORE_GAME_SLUG = {
    "League of Legends": "lol",
    "Dota 2": "dota2",
    "CS2": "csgo",
    "Valorant": "valorant",
}


def diagnosticar_equipo_en_pandascore(team_name, headers, game=None):
    """
    DIAGNOSTICO TEMPORAL (2026-07-28 v2, 2026-07-29 v3): cuando
    /matches?search[name]=equipo da 0 resultados, esto puede ser porque el
    equipo no esta en la base de PandaScore, porque "search[name]" en
    /matches filtra sobre el NOMBRE DEL PARTIDO (no del equipo), o porque el
    endpoint generico /teams no cubre bien datos por-juego (PandaScore separa
    mucho por videojuego). v3 agrega un segundo chequeo contra el endpoint
    especifico del juego (ej. /lol/teams) cuando el generico /teams tambien
    da 0, ya que /teams generico dio 0 incluso para 'T1' en la corrida del
    2026-07-29, lo cual descarta la hipotesis original de v2. Quitar una vez
    confirmada la causa real.
    """
    query = urllib.parse.urlencode({"search[name]": team_name, "per_page": 5})
    url = f"{PANDASCORE_TEAMS_URL}?{query}"
    data = fetch_json(url, headers=headers)
    if data is None:
        print(f"  [DIAG-TEAMS] '{team_name}': fetch_json devolvio None en /teams", file=sys.stderr)
    elif not isinstance(data, list):
        print(f"  [DIAG-TEAMS] '{team_name}': /teams respuesta NO es lista (tipo={type(data).__name__}): {str(data)[:300]}", file=sys.stderr)
    elif len(data) == 0:
        print(f"  [DIAG-TEAMS] '{team_name}': /teams (generico) da 0 resultados", file=sys.stderr)
    else:
        resumen = [
            {"id": t.get("id"), "name": t.get("name"), "slug": t.get("slug"),
             "videogame": (t.get("current_videogame") or {}).get("slug") if isinstance(t.get("current_videogame"), dict) else None}
            for t in data
        ]
        print(f"  [DIAG-TEAMS] '{team_name}': /teams SI encuentra {len(data)} equipo(s) -> {resumen}", file=sys.stderr)
        print(f"  [DIAG-TEAMS] '{team_name}': esto confirma que el equipo existe en PandaScore; "
              f"el problema esta en como /matches?search[name] filtra, no en falta de datos", file=sys.stderr)
        return

    slug = PANDASCORE_GAME_SLUG.get(game)
    if not slug:
        print(f"  [DIAG-GAME] '{team_name}': sin slug de juego conocido para '{game}', se omite chequeo por-juego", file=sys.stderr)
        return

    url_juego = f"https://api.pandascore.co/{slug}/teams?{query}"
    data_juego = fetch_json(url_juego, headers=headers)
    if data_juego is None:
        print(f"  [DIAG-GAME] '{team_name}': fetch_json devolvio None en /{slug}/teams", file=sys.stderr)
    elif not isinstance(data_juego, list):
        print(f"  [DIAG-GAME] '{team_name}': /{slug}/teams respuesta NO es lista (tipo={type(data_juego).__name__}): {str(data_juego)[:300]}", file=sys.stderr)
    elif len(data_juego) == 0:
        print(f"  [DIAG-GAME] '{team_name}': /{slug}/teams (especifico del juego) TAMBIEN da 0 -> "
              f"probablemente problema de alcance/plan de la API key, no de endpoint", file=sys.stderr)
    else:
        resumen = [{"id": t.get("id"), "name": t.get("name"), "slug": t.get("slug")} for t in data_juego]
        print(f"  [DIAG-GAME] '{team_name}': /{slug}/teams SI encuentra {len(data_juego)} equipo(s) -> {resumen}", file=sys.stderr)
        print(f"  [DIAG-GAME] '{team_name}': el endpoint generico /teams no sirve para este juego, "
              f"hay que usar /{slug}/teams (y por extension /{slug}/matches) en el bot", file=sys.stderr)


def fetch_pandascore_matches(team_name, game=None):
    """Busca partidos recientes/proximos de un equipo. None si falla o vacio."""
    query = urllib.parse.urlencode({"search[name]": team_name, "per_page": 20})
    url = f"{PANDASCORE_MATCHES_URL}?{query}"
    headers = {"User-Agent": "esports_bot/1.0"}
    api_key = os.environ.get("PANDASCORE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = fetch_json(url, headers=headers)

    # DIAGNOSTICO TEMPORAL (2026-07-28): el bot nunca encuentra datos para
    # ningun equipo y fetch_json no imprime nada en el caso "200 pero vacio",
    # asi que no hay forma de saber si falta la API key, si PandaScore
    # devuelve una lista vacia, o un cuerpo de error no-lista. Quitar estas
    # lineas una vez confirmada la causa real.
    if not api_key:
        print(f"  [DIAG] PANDASCORE_API_KEY no esta configurada (llamando sin autenticacion)", file=sys.stderr)
    if data is None:
        print(f"  [DIAG] '{team_name}': fetch_json devolvio None (ver [HTTP]/[URLError]/[Error] arriba)", file=sys.stderr)
    elif isinstance(data, list):
        print(f"  [DIAG] '{team_name}': respuesta lista con {len(data)} elemento(s)", file=sys.stderr)
    else:
        print(f"  [DIAG] '{team_name}': respuesta NO es una lista (tipo={type(data).__name__}): {str(data)[:300]}", file=sys.stderr)

    if not data or not isinstance(data, list) or len(data) == 0:
        diagnosticar_equipo_en_pandascore(team_name, headers, game=game)
        return None
    return data


def _es_ganador(match, team_name_lower):
    winner = match.get("winner") or {}
    winner_name = str(winner.get("name", "")).lower() if isinstance(winner, dict) else ""
    if winner_name:
        return team_name_lower in winner_name or winner_name in team_name_lower
    return False


def _oponentes(match):
    nombres = []
    for op in match.get("opponents", []) or []:
        opp = op.get("opponent") if isinstance(op, dict) else None
        if isinstance(opp, dict) and opp.get("name"):
            nombres.append(str(opp["name"]))
    return nombres


def calcular_forma_reciente(matches, team_name):
    """Win rate en los ultimos 10 partidos finalizados. 0.5 si no hay datos."""
    team_lower = team_name.lower()
    finalizados = [m for m in matches if m.get("status") == "finished"]
    if not finalizados:
        return 0.5
    finalizados = finalizados[:10]
    victorias = sum(1 for m in finalizados if _es_ganador(m, team_lower))
    return victorias / len(finalizados)


def calcular_h2h(matches, team_a, team_b):
    """Win rate de team_a en partidos contra team_b especificamente."""
    team_a_lower = team_a.lower()
    team_b_lower = team_b.lower()
    enfrentamientos = []
    for m in matches:
        if m.get("status") != "finished":
            continue
        oponentes = [n.lower() for n in _oponentes(m)]
        if any(team_b_lower in n or n in team_b_lower for n in oponentes):
            enfrentamientos.append(m)
    if not enfrentamientos:
        return 0.5
    victorias = sum(1 for m in enfrentamientos if _es_ganador(m, team_a_lower))
    return victorias / len(enfrentamientos)


def _texto_torneo(match):
    partes = []
    for key in ("league", "serie", "tournament"):
        obj = match.get(key)
        if isinstance(obj, dict):
            partes.append(str(obj.get("name", "")))
            partes.append(str(obj.get("full_name", "")))
    return " ".join(partes).lower()


def calcular_tier(matches, team_b):
    """Nivel del torneo del proximo/actual enfrentamiento contra team_b."""
    team_b_lower = team_b.lower()
    for m in matches:
        oponentes = [n.lower() for n in _oponentes(m)]
        if not any(team_b_lower in n or n in team_b_lower for n in oponentes):
            continue
        texto = _texto_torneo(m)
        if "world" in texto or "worlds" in texto or "championship" in texto:
            return 1.0
        if "major" in texto or "msi" in texto:
            return 0.66
        return 0.33
    return 0.5


def calcular_region(matches, team_name, team_b):
    """Ventaja de 'jugar en casa': region del equipo vs region del torneo."""
    region_equipo = EQUIPO_REGION.get(team_name.lower().strip())
    if not region_equipo:
        return 0.5

    team_b_lower = team_b.lower()
    for m in matches:
        oponentes = [n.lower() for n in _oponentes(m)]
        if not any(team_b_lower in n or n in team_b_lower for n in oponentes):
            continue
        texto = _texto_torneo(m)
        keywords = REGION_KEYWORDS.get(region_equipo, [])
        if any(k in texto for k in keywords):
            return 1.0
        return 0.0
    return 0.5


def calcular_probabilidad(team_a, team_b, game=None):
    """
    Calcula nuestra probabilidad de victoria de team_a contra team_b.
    Devuelve (probabilidad, tiene_datos).
    Si PandaScore falla o no hay datos, cae a 50/50 y no calcula edge.
    """
    matches_a = fetch_pandascore_matches(team_a, game=game)
    if not matches_a:
        return 0.5, False

    forma_reciente = calcular_forma_reciente(matches_a, team_a)
    h2h = calcular_h2h(matches_a, team_a, team_b)
    tier = calcular_tier(matches_a, team_b)
    region = calcular_region(matches_a, team_a, team_b)

    probabilidad = (
        forma_reciente * 0.40
        + h2h * 0.30
        + tier * 0.20
        + region * 0.10
    )
    return probabilidad, True


# ----------------------------------------------------------------------
# Decision de apuesta
# ----------------------------------------------------------------------

def decidir(our_probability, market_price, tiene_datos):
    """Devuelve (action, side, edge) segun EDGE_MINIMO."""
    if not tiene_datos:
        return "SKIP", None, 0.0

    edge = our_probability - market_price
    if edge >= EDGE_MINIMO:
        return "BET", "YES", edge
    if edge <= -EDGE_MINIMO:
        return "BET", "NO", abs(edge)
    return "SKIP", None, edge


# ----------------------------------------------------------------------
# Ejecucion real (py-clob-client)
# ----------------------------------------------------------------------

def colocar_orden_real(market, side, usd_amount):
    """Intenta colocar una orden real via py-clob-client. Devuelve dict de resultado."""
    api_key = os.environ.get("API_KEY")
    api_secret = os.environ.get("API_SECRET")
    api_passphrase = os.environ.get("API_PASSPHRASE")
    private_key = os.environ.get("PRIVATE_KEY")

    if not all([api_key, api_secret, api_passphrase, private_key]):
        return {"ok": False, "error": "faltan variables de entorno para modo real"}

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds, OrderArgs
        from py_clob_client.order_builder.constants import BUY
    except ImportError as e:
        return {"ok": False, "error": f"py-clob-client no disponible: {e}"}

    try:
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            creds=creds,
        )

        token_ids = market.get("token_ids") or []
        outcomes = market.get("outcomes") or []
        idx = 0
        if side == "NO" and len(outcomes) > 1:
            idx = 1
        if idx >= len(token_ids):
            return {"ok": False, "error": "no se encontro token_id para el side elegido"}
        token_id = token_ids[idx]

        price = market["yes_price"] if side == "YES" else round(1 - market["yes_price"], 4)
        size = round(usd_amount / price, 2) if price > 0 else 0

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY,
        )
        signed_order = client.create_order(order_args)
        response = client.post_order(signed_order)
        return {"ok": True, "response": response}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

def registrar(entry):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------
# Ciclo principal
# ----------------------------------------------------------------------

def ejecutar_ciclo(modo_real):
    ahora_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"=== esports_bot.py :: ciclo {ahora_iso} ===")

    mercados = fetch_esports_markets()
    print(f"Mercados de esports encontrados: {len(mercados)}")

    if not mercados:
        registrar({
            "timestamp": ahora_iso,
            "action": "NO_MARKET",
            "market": None,
            "game": None,
            "team_bet": None,
            "side": None,
            "our_prob": None,
            "market_price": None,
            "edge": None,
            "bet_usd": 0.0,
            "liquidity": None,
            "volume24h": None,
            "market_id": None,
            "slug": None,
        })
        print("No se encontraron mercados de esports que cumplan los filtros.")
        return

    resumen = []
    for market in mercados:
        equipos = extraer_equipos(market["question"])
        if not equipos:
            print(f"  [SKIP] No se pudieron extraer equipos de: {market['question']}")
            continue
        team_a, team_b = equipos

        our_probability, tiene_datos = calcular_probabilidad(team_a, team_b, game=market["juego"])
        action, side, edge = decidir(our_probability, market["yes_price"], tiene_datos)

        team_bet = None
        if action == "BET":
            team_bet = team_a if side == "YES" else team_b

        entry = {
            "timestamp": ahora_iso,
            "action": action,
            "market": market["question"],
            "game": market["juego"],
            "team_bet": team_bet,
            "side": side,
            "our_prob": round(our_probability, 4),
            "market_price": round(market["yes_price"], 4),
            "edge": round(edge, 4),
            "bet_usd": BET_USD if action == "BET" else 0.0,
            "liquidity": market["liquidity"],
            "volume24h": market["volume24h"],
            "market_id": market["id"],
            "slug": market["slug"],
        }

        if action == "BET":
            if modo_real:
                resultado = colocar_orden_real(market, side, BET_USD)
                entry["real_mode"] = True
                entry["real_result"] = resultado
                if resultado.get("ok"):
                    print(f"  [REAL] Orden colocada: {market['question']} -> {side} (edge {edge:.2%})")
                else:
                    print(f"  [REAL-FALLO] {resultado.get('error')} -- cae a SIMULACION")
            else:
                print(f"  [SIMULACION] BET {side} en '{market['question']}' "
                      f"(our_prob={our_probability:.2f}, mkt={market['yes_price']:.2f}, edge={edge:.2%})")
        else:
            print(f"  [SKIP] '{market['question']}' (our_prob={our_probability:.2f}, "
                  f"mkt={market['yes_price']:.2f}, edge={edge:.2%}, datos={tiene_datos})")

        registrar(entry)
        resumen.append(entry)

    apuestas = [e for e in resumen if e["action"] == "BET"]
    print("--- Resumen del ciclo ---")
    print(f"Total evaluados: {len(resumen)} | Apuestas: {len(apuestas)} | "
          f"Omitidos: {len(resumen) - len(apuestas)}")
    for e in apuestas:
        print(f"  * {e['market']} -> {e['side']} (edge {e['edge']:.2%}, ${e['bet_usd']})")


def main():
    parser = argparse.ArgumentParser(description="Bot de esports para Polymarket")
    parser.add_argument("--real", action="store_true", help="Ejecuta ordenes reales via py-clob-client")
    args = parser.parse_args()

    try:
        ejecutar_ciclo(modo_real=args.real)
    except Exception as e:
        print(f"[ERROR FATAL] El ciclo fallo de forma inesperada: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
