"""Cluster bootstrap by contract: questions within one contract are not independent."""

from __future__ import annotations

import numpy as np
import pandas as pd


def boot_by_group(values, groups, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of the pooled mean, resampling whole groups (NaNs ignored)."""
    v = np.asarray(values, float)
    g = pd.factorize(np.asarray(groups))[0]
    k = g.max() + 1
    ok = ~np.isnan(v)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    sums, cnts = np.bincount(g[ok], v[ok], minlength=k), np.bincount(g[ok], minlength=k)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        out[b] = (w * sums).sum() / max((w * cnts).sum(), 1)
    return out


def ci(a) -> tuple[float, float]:
    lo, hi = np.nanpercentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def rate(df: pd.DataFrame, col: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean of ``df[col]`` with a 95% bootstrap interval over ``df.doc``."""
    v = df[col].astype(float)
    b = boot_by_group(v.values, df.doc.values, n_boot, seed)
    return (float(np.nanmean(v)), *ci(b))


def paired(df_a: pd.DataFrame, df_b: pd.DataFrame, col: str, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean paired difference A - B over shared ``qid``s, bootstrapped over contracts."""
    a = df_a.set_index("qid")[col].astype(float)
    b = df_b.set_index("qid")[col].astype(float)
    j = a.index.intersection(b.index)
    d = (a[j] - b[j]).values
    g = df_a.set_index("qid").loc[j, "doc"].values
    return (float(np.nanmean(d)), *ci(boot_by_group(d, g, n_boot, seed)))
