"""Legacy CLI wrapper for TSMOM/DMN backtests.

Prefer using: `python -m dmn.cli`
"""

from __future__ import annotations

from dmn.cli.main import run


if __name__ == "__main__":
    raise SystemExit(run())
