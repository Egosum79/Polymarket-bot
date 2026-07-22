#!/usr/bin/env python3
"""
=======================================================
  POLYMARKET BOT — REVISIÓN DIARIA
  Lee bot_log.jsonl y publica reporte en GitHub Issues
=======================================================

Uso (llamado automáticamente por GitHub Actions):
    python daily_review.py

Variables de entorno requeridas (configuradas por GitHub):
    GITHUB_TOKEN   — token automático de GitHub Actions
    GITHUB_REPO    — owner/repo (ej: Egosum79/Polymarket-bot)
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────
# CARGA DEL LOG
# ─────────────────────────────────────────────────────

def load_log(path: str = "bot_log.jsonl") -> list[dict]:
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


def entries_last_24h(entries: list[dict]) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    result = []
    for e in entries:
        try:
            ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            if ts >= cutoff:
                result.append(e)
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────
# ANÁLISIS DEL RESUMEN
# ─────────────────────────────────────────────────────

def analyze(entries: list[dict]) -> dict:
    total      = len(entries)
    long_yes   = [e for e in entries if e.get("bet_side") == "YES"]
    long_no    = [e for e in entries if e.get("bet_side") == "NO"]

    # Agrupar por mercado (pregunta)
    by_market  = defaultdict(list)
    for e in entries:
        by_market[e.get("question", "?")].append(e)

    # Señal más fuerte del día
    top = sorted(entries, key=lambda x: abs(x.get("edge", 0)), reverse=True)[:5]

    # Promedio de edge
    edges = [abs(e.get("edge", 0)) for e in entries if e.get("edge")]
    avg_edge = sum(edges) / len(edges) if edges else 0

    # Apuesta total simulada
    total_bet = sum(e.get("bet_usd", 0) for e in entries)

    return {
        "total":      total,
        "long_yes":   len(long_yes),
        "long_no":    len(long_no),
        "markets":    len(by_market),
        "avg_edge":   avg_edge,
        "total_bet":  total_bet,
        "top":        top,
    }


def analyze_btc(entries: list[dict]) -> dict:
    """Analiza entradas del btc_direction_bot (btc_bot_log.jsonl)."""
    total     = len(entries)
    bets_up   = [e for e in entries if e.get("action") == "BET" and e.get("direction") == "UP"]
    bets_down = [e for e in entries if e.get("action") == "BET" and e.get("direction") == "DOWN"]
    skipped   = [e for e in entries if e.get("action") == "SKIP"]
    no_market = [e for e in entries if e.get("action") in ("NO_MARKET", "PASS")]

    edges = [abs(e.get("edge", 0)) for e in entries if e.get("action") == "BET" and e.get("edge")]
    avg_edge = sum(edges) / len(edges) if edges else 0

    total_bet = sum(e.get("bet_usd", 0) for e in entries if e.get("action") == "BET")

    # Top señales por edge
    top = sorted(
        [e for e in entries if e.get("action") == "BET"],
        key=lambda x: abs(x.get("edge", 0)),
        reverse=True
    )[:3]

    return {
        "total":      total,
        "bets_up":    len(bets_up),
        "bets_down":  len(bets_down),
        "skipped":    len(skipped),
        "no_market":  len(no_market),
        "avg_edge":   avg_edge,
        "total_bet":  total_bet,
        "top":        top,
    }


# ─────────────────────────────────────────────────────
# GENERADOR DEL REPORTE
# ─────────────────────────────────────────────────────

def build_report(summary: dict, all_entries: list[dict],
                 btc_summary: dict, all_btc_entries: list[dict]) -> tuple[str, str]:
    """Retorna (título, cuerpo) del issue de GitHub."""
    now    = datetime.now(timezone.utc)
    today  = now.strftime("%Y-%m-%d")
    hora   = now.strftime("%H:%M UTC")

    title = f"📊 Reporte Diario Bot — {today}"

    total_all     = len(all_entries)
    total_all_btc = len(all_btc_entries)

    lines = [
        f"## 🤖 Reporte Diario — {today}",
        f"*Generado automáticamente a las {hora}*",
        "",
        "---",
        "",
        "## 📈 Bot 1: Polymarket Señales (bot_log.jsonl)",
        "",
        "### Señales detectadas (últimas 24h)",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| Total ciclos/señales | {summary['total']} |",
        f"| 🟢 LONG YES | {summary['long_yes']} |",
        f"| 🔴 LONG NO | {summary['long_no']} |",
        f"| Mercados únicos | {summary['markets']} |",
        f"| Edge promedio | {summary['avg_edge']*100:.1f}% |",
        f"| Apuesta simulada total | ${summary['total_bet']:.2f} |",
        f"| Total histórico en log | {total_all} |",
        "",
    ]

    if summary["top"]:
        lines += [
            "#### 🎯 Top señales del día",
            "",
        ]
        for i, s in enumerate(summary["top"], 1):
            side_emoji = "🟢" if s.get("bet_side") == "YES" else "🔴"
            link = f"https://polymarket.com/event/{s.get('slug','')}"
            lines += [
                f"**{i}. {side_emoji} BET {s.get('bet_side')}** — Edge: {s.get('edge',0)*100:.1f}% — ${s.get('bet_usd',0):.2f}",
                f"> {s.get('question','')[:100]}",
                f"> BTC: ${s.get('btc_price',0):,.0f} → Target: ${s.get('target_price',0):,.0f} | "
                f"Mercado: {int(s.get('yes_price',0)*100)}¢ YES | Nuestra estima: {int(s.get('our_prob',0)*100)}¢",
                f"> [Ver en Polymarket]({link})",
                "",
            ]
    else:
        lines += [
            "#### ⚪ Sin señales fuertes en las últimas 24h",
            "",
            "El modelo no encontró oportunidades con edge ≥ 8% en este período.",
            "",
        ]

    # ── Sección BTC Direction Bot ──────────────────────────────────────────
    lines += [
        "---",
        "",
        "## ₿ Bot 2: BTC Dirección 1H (btc_bot_log.jsonl)",
        "",
        "### Ciclos (últimas 24h)",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| Total ciclos | {btc_summary['total']} |",
        f"| 🟢 Apuestas UP | {btc_summary['bets_up']} |",
        f"| 🔴 Apuestas DOWN | {btc_summary['bets_down']} |",
        f"| ⏭️ Sin edge suficiente (SKIP) | {btc_summary['skipped']} |",
        f"| 🔍 Sin mercado disponible | {btc_summary['no_market']} |",
        f"| Edge promedio (apuestas) | {btc_summary['avg_edge']*100:.1f}% |",
        f"| Apuesta simulada total | ${btc_summary['total_bet']:.2f} |",
        f"| Total histórico en log | {total_all_btc} |",
        "",
    ]

    if btc_summary["top"]:
        lines += [
            "#### 🎯 Mejores apuestas del día",
            "",
        ]
        for i, s in enumerate(btc_summary["top"], 1):
            direction = s.get("direction", "?")
            dir_emoji = "🟢" if direction == "UP" else "🔴"
            lines += [
                f"**{i}. {dir_emoji} {direction}** — Edge: {s.get('edge',0)*100:.1f}% — "
                f"Nuestra prob: {s.get('our_prob',0)*100:.0f}% | Mercado: {s.get('market_price',0)*100:.0f}¢",
                f"> {s.get('market','')[:120]}",
                "",
            ]
    else:
        lines += [
            "#### ⚪ Sin apuestas en las últimas 24h",
            "",
            "El bot no encontró oportunidades con edge ≥ 6% en este período.",
            "",
        ]

    lines += [
        "---",
        "",
        "*⚠️ Este reporte es informativo. No constituye asesoría financiera.*",
        "*Ambos bots operan en modo SIMULACIÓN — no se ejecutan apuestas reales.*",
    ]

    return title, "\n".join(lines)


# ─────────────────────────────────────────────────────
# PUBLICAR EN GITHUB ISSUES
# ─────────────────────────────────────────────────────

def create_github_issue(title: str, body: str) -> bool:
    token = os.environ.get("GITHUB_TOKEN")
    repo  = os.environ.get("GITHUB_REPO")   # ej: Egosum79/Polymarket-bot

    if not token or not repo:
        print("⚠️  GITHUB_TOKEN o GITHUB_REPO no configurados")
        print("   Mostrando reporte en consola:\n")
        print(f"# {title}\n")
        print(body)
        return False

    url     = f"https://api.github.com/repos/{repo}/issues"
    payload = json.dumps({
        "title":  title,
        "body":   body,
        "labels": ["bot-report"],
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data    = payload,
        method  = "POST",
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github+json",
            "Content-Type":  "application/json",
            "User-Agent":    "polymarket-bot",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            issue_url = result.get("html_url", "")
            print(f"✅ Issue creado: {issue_url}")
            return True
    except urllib.error.HTTPError as e:
        body_err = e.read().decode()
        print(f"❌ Error creando issue: {e.code} — {body_err}")
        return False


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  POLYMARKET BOT — REVISIÓN DIARIA")
    print("=" * 60)

    # Bot 1: Polymarket señales
    all_entries    = load_log("bot_log.jsonl")
    recent_entries = entries_last_24h(all_entries)

    print(f"\n📋 [Bot 1] Entradas totales: {len(all_entries)}")
    print(f"📋 [Bot 1] Últimas 24h:      {len(recent_entries)}")

    summary = analyze(recent_entries)

    # Bot 2: BTC dirección 1H
    all_btc_entries    = load_log("btc_bot_log.jsonl")
    recent_btc_entries = entries_last_24h(all_btc_entries)

    print(f"\n₿  [Bot 2] Entradas totales: {len(all_btc_entries)}")
    print(f"₿  [Bot 2] Últimas 24h:      {len(recent_btc_entries)}")

    btc_summary = analyze_btc(recent_btc_entries)

    title, body = build_report(summary, all_entries, btc_summary, all_btc_entries)

    print(f"\n📝 Publicando reporte: '{title}'")
    create_github_issue(title, body)

    print("\n✅ Revisión diaria completada.")


if __name__ == "__main__":
    main()
