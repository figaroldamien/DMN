from __future__ import annotations

import pandas as pd


def compare_cleaners(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float]:
    # This stays intentionally simple: for the RIE step we mainly want a small,
    # deterministic regression harness against a trusted external implementation.
    diff = (reference - candidate).to_numpy(dtype=float)
    return {
        "max_abs_diff": float(abs(diff).max()),
        "mean_abs_diff": float(abs(diff).mean()),
    }
