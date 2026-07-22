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
import smtplib
import ssl
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from collections import defaultdict

# En consolas Windows con codepage legado (cp1252), imprimir emojis revienta
# con UnicodeEncodeError. Forzamos stdout/stderr a UTF-8 si el terminal lo permite.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Email (modo REAL de entrega — SMTP con contraseña de aplicación) ──
# Configurar como variables de entorno (o secretos de GitHub Actions):
#   SMTP_HOST  (default: smtp.office365.com)
#   SMTP_PORT  (default: 587)
#   SMTP_USER  — cuenta remitente
#   SMTP_PASS  — contraseña de aplicación (NUNCA la contraseña normal de la cuenta)
#   REPORT_TO  — destinatario (default: hectorhugocortez@hotmail.com)
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.office365.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
REPORT_TO = os.environ.get("REPORT_TO", "hectorhugocortez@hotmail.com")


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
    bets_up   = [e for e in entries if e.get("action") == "BET" and e.get("bet_side") == "UP"]
    bets_down = [e for e in entries if e.get("action") == "BET" and e.get("bet_side") == "DOWN"]
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
            direction = s.get("bet_side", "?")
            dir_emoji = "🟢" if direction == "UP" else "🔴"
            lines += [
                f"**{i}. {dir_emoji} {direction}** — Edge: {s.get('edge',0)*100:.1f}% — "
                f"Nuestra prob: {s.get('our_prob',0)*100:.0f}% | Mercado: {s.get('market_prob',0)*100:.0f}¢",
                f"> {s.get('market_q','')[:120]}",
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
# GENERADOR DEL REPORTE (HTML PARA EMAIL)
# ─────────────────────────────────────────────────────

def _table_row(cells: list[str]) -> str:
    tds = "".join(f"<td style='padding:4px 10px;border-bottom:1px solid #eee'>{c}</td>" for c in cells)
    return f"<tr>{tds}</tr>"


def build_email_html(summary: dict, all_entries: list[dict],
                      btc_summary: dict, all_btc_entries: list[dict]) -> str:
    now   = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    hora  = now.strftime("%H:%M UTC")

    def metrics_table(rows: list[tuple[str, str]]) -> str:
        body = "".join(_table_row([k, v]) for k, v in rows)
        return (f"<table style='border-collapse:collapse;font-family:Arial,sans-serif;"
                f"font-size:14px;width:100%;max-width:480px'>{body}</table>")

    parts = [
        f"<div style='font-family:Arial,sans-serif;color:#222'>",
        f"<h2>🤖 Reporte Diario — {today}</h2>",
        f"<p style='color:#666'>Generado automáticamente a las {hora}</p>",
        "<hr>",
        "<h3>📈 Bot 1: Polymarket Señales</h3>",
        metrics_table([
            ("Total ciclos/señales", str(summary['total'])),
            ("🟢 LONG YES", str(summary['long_yes'])),
            ("🔴 LONG NO", str(summary['long_no'])),
            ("Mercados únicos", str(summary['markets'])),
            ("Edge promedio", f"{summary['avg_edge']*100:.1f}%"),
            ("Apuesta simulada total", f"${summary['total_bet']:.2f}"),
            ("Total histórico en log", str(len(all_entries))),
        ]),
    ]

    if summary["top"]:
        parts.append("<h4>🎯 Top señales del día</h4><ul>")
        for s in summary["top"]:
            side_emoji = "🟢" if s.get("bet_side") == "YES" else "🔴"
            link = f"https://polymarket.com/event/{s.get('slug','')}"
            parts.append(
                f"<li>{side_emoji} <b>BET {s.get('bet_side')}</b> — Edge: {s.get('edge',0)*100:.1f}% "
                f"— ${s.get('bet_usd',0):.2f}<br>"
                f"<span style='color:#555'>{s.get('question','')[:100]}</span><br>"
                f"BTC ${s.get('btc_price',0):,.0f} → Target ${s.get('target_price',0):,.0f} | "
                f"Mercado {int(s.get('yes_price',0)*100)}¢ YES | Nuestra estima {int(s.get('our_prob',0)*100)}¢ "
                f"— <a href='{link}'>Ver en Polymarket</a></li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>⚪ Sin señales fuertes en las últimas 24h (edge insuficiente).</p>")

    parts += [
        "<hr>",
        "<h3>₿ Bot 2: BTC Dirección 1H</h3>",
        metrics_table([
            ("Total ciclos", str(btc_summary['total'])),
            ("🟢 Apuestas UP", str(btc_summary['bets_up'])),
            ("🔴 Apuestas DOWN", str(btc_summary['bets_down'])),
            ("⏭️ Sin edge suficiente (SKIP)", str(btc_summary['skipped'])),
            ("🔍 Sin mercado disponible", str(btc_summary['no_market'])),
            ("Edge promedio (apuestas)", f"{btc_summary['avg_edge']*100:.1f}%"),
            ("Apuesta simulada total", f"${btc_summary['total_bet']:.2f}"),
            ("Total histórico en log", str(len(all_btc_entries))),
        ]),
    ]

    if btc_summary["top"]:
        parts.append("<h4>🎯 Mejores apuestas del día</h4><ul>")
        for s in btc_summary["top"]:
            direction = s.get("bet_side", "?")
            dir_emoji = "🟢" if direction == "UP" else "🔴"
            parts.append(
                f"<li>{dir_emoji} <b>{direction}</b> — Edge: {s.get('edge',0)*100:.1f}% — "
                f"Nuestra prob: {s.get('our_prob',0)*100:.0f}% | Mercado: {s.get('market_prob',0)*100:.0f}¢<br>"
                f"<span style='color:#555'>{s.get('market_q','')[:120]}</span></li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>⚪ Sin apuestas en las últimas 24h.</p>")

    parts += [
        "<hr>",
        "<p style='color:#999;font-size:12px'>⚠️ Este reporte es informativo. No constituye asesoría "
        "financiera. Ambos bots operan en modo SIMULACIÓN — no se ejecutan apuestas reales.</p>",
        "</div>",
    ]
    return "\n".join(parts)


# ─────────────────────────────────────────────────────
# ENVÍO POR CORREO (SMTP)
# ─────────────────────────────────────────────────────

def send_email_report(title: str, html_body: str) -> bool:
    if not SMTP_USER or not SMTP_PASS:
        print("⚠️  SMTP_USER / SMTP_PASS no configurados — no se puede enviar el correo")
        print("   Configura estas variables de entorno (o secretos de GitHub Actions) para habilitar el envío.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = title
    msg["From"] = SMTP_USER
    msg["To"] = REPORT_TO
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [REPORT_TO], msg.as_string())
        print(f"✅ Correo enviado a {REPORT_TO}")
        return True
    except Exception as e:
        print(f"❌ Error enviando correo: {e}")
        return False


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
    html_body   = build_email_html(summary, all_entries, btc_summary, all_btc_entries)

    print(f"\n📧 Enviando reporte por correo a {REPORT_TO}...")
    send_email_report(title, html_body)

    print(f"\n📝 Publicando reporte en GitHub Issues: '{title}'")
    create_github_issue(title, body)

    print("\n✅ Revisión diaria completada.")


if __name__ == "__main__":
    main()
