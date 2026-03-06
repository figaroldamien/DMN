"""Legacy CLI wrapper for TSMOM/DMN backtests.

Prefer using: `python -m dmn.cli`
"""

from __future__ import annotations

from dmn.cli.main import run

# Exemples de lancement (CLI directe)
# -----------------------------------
# Lancement simple:
# python -m dmn.cli --market cac40 --start 2010-01-01
#
# Lancement avec un ticker unique (au lieu de --market):
# python -m dmn.cli --ticker AAPL --start 2015-01-01
#
# Filtre par secteur / sous-secteur:
# python -m dmn.cli --market nasdaq100 --sector Technology --sub-sector Software
#
# Activer ML, désactiver DMN:
# python -m dmn.cli --market cac40 --run-ml --no-run-dmn
#
# Changer les paramètres de backtest:
# python -m dmn.cli --market table8_all --sigma-target-annual 0.12 --vol-span 40 --cost-bps 1.5 --min-obs 300
#
# Par defaut la configuration effective est affichee.
# Pour la masquer:
# python -m dmn.cli --market cac40 --no-print-config
#
# Lancement avec fichier de configuration:
# python -m dmn.cli --config src/dmn/cli/config.example.toml
#
# Fichier de configuration exemple (TOML):
# ----------------------------------------
# market = "cac40"
# ticker = "AAPL" # Utiliser ticker OU market, pas les deux.
# start = "2000-01-01"
# sector = "Financials"
# sub_sector = "Banks"
# run_ml = false
# run_dmn = true
#
# [backtest]
# sigma_target_annual = 0.15
# vol_span = 60
# cost_bps = 2.0
# portfolio_vol_target = true
# min_obs = 400


if __name__ == "__main__":
    raise SystemExit(run(["--market", "dataset_all", "--start", "2000-01-01", "--no-run-ml", "--run-dmn"]))
