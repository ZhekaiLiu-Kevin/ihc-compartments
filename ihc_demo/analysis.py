"""Summaries compare measured values with independently drawn synthetic targets."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .measurement import quantify

SCALES = (1.25, 1.5, 1.75, 2.0)
DEFAULT_SEEDS = (17, 31, 47, 59, 73)


def ring_sweep(rgb, labels, truth, baseline=None):
    """Measure the same fixed nuclear labels at several outer radii."""
    runs = []
    for scale in SCALES:
        cells = baseline if scale == 1.5 and baseline is not None else quantify(
            rgb, labels, outer_scale=scale
        )[0]
        values = cells[["cell_id", "ring_mean_dab_od", "ring_to_nucleus_ratio"]].copy()
        values["outer_scale"] = scale
        runs.append(values.merge(truth[["cell_id", "phenotype"]], on="cell_id", validate="one_to_one"))
    return pd.concat(runs, ignore_index=True)


def recovery(cells, truth):
    """Return one record per compartment, using only paired finite measurements.

    A target is the simulation's input concentration, not the expected recovered
    mean after texture, edge smoothing and geometry mismatch.
    """
    data = cells.merge(truth, on="cell_id", validate="one_to_one")
    summary = []
    for compartment in ("nuclear", "ring"):
        target = data[f"target_{compartment}_dab_od"].to_numpy()
        measured = data[f"{compartment}_mean_dab_od"].to_numpy()
        ok = np.isfinite(target) & np.isfinite(measured)
        residual = measured[ok] - target[ok]
        n = int(ok.sum())
        variable = n >= 3 and np.unique(target[ok]).size > 1 and np.unique(measured[ok]).size > 1
        summary.append({
            "compartment": compartment, "cells_total": len(data), "cells_measured": n,
            "rank_correlation": float(spearmanr(target[ok], measured[ok]).statistic) if variable else np.nan,
            "mean_absolute_error_od": float(np.mean(np.abs(residual))) if n else np.nan,
            "mean_signed_error_od": float(np.mean(residual)) if n else np.nan,
        })
    return pd.DataFrame(summary)
