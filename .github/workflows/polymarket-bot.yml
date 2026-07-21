name: Polymarket Bot

on:
  schedule:
    - cron: '0 * * * *'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-bot:
    runs-on: ubuntu-latest

    steps:
      - name: Descargar repositorio
        uses: actions/checkout@v4

      - name: Configurar Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Ver archivos disponibles
        run: ls -la

      - name: Correr bot (1 ciclo)
        run: python polymarket_bot.py --ciclos 1

      - name: Guardar log en el repositorio
        run: |
          git config --global user.email "polymarket-bot@github"
          git config --global user.name "Polymarket Bot"
          git add bot_log.jsonl || true
          git diff --cached --quiet || git commit -m "Bot cycle $(date '+%Y-%m-%d %H:%M UTC')"
          git push || true
