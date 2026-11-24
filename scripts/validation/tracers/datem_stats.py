"""DATEM statistics and ranking score: a re-implementation of ``statmain``.

Follows ``stat_pgm/statmain/statmain.f`` (DATEM, revised 31 Oct 2012)
routine by routine, including its quirks, so that it reproduces the posted
``statA?_000.txt`` and ``statT?_000.txt`` files:

- a pair is missing when either value is negative; measured values at or
  below the zero level are set to 0 (``-z``); with ``-y`` the calculated
  values are too;
- ``-x`` drops zero–zero pairs, ``-p`` keeps only pairs with both > 0;
- percentiles are ``sorted[max(1, int(i*n/100.0)) - 1]`` (single precision);
- the Kolmogorov–Smirnov parameter compares the 5th–95th percentiles of the
  two distributions and is in percent;
- FAR, POD and TS use the contingency level (``-l``, default 1) on the raw
  values, missing pairs included, as ``statmain`` does;
- the resampled correlation (``-b``) makes N passes (N is the number of
  records), each dropping every pair independently with probability 0.2,
  and reports the root-mean-square of the N correlations. ``statmain`` takes
  its random numbers from the compiler's generator; here a seed is given.

Arithmetic is single precision with sequential sums, as in ``statmain``
(``REAL``), because double precision changes the last printed digit of a
few values (factor-of-5 counts at ratio boundaries, averages in ``-at``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

F = np.float32


def _ssum(x) -> np.float32:
    """Sequential single-precision sum, as a Fortran DO loop accumulates."""
    x = np.asarray(x, dtype=F)
    return np.cumsum(x, dtype=F)[-1] if x.size else F(0.0)


KEYS = ["yr", "mo", "dy", "hhmm", "dur_hhmm", "site_num"]


@dataclass(frozen=True)
class StatOptions:
    """``statmain`` options that change the scores (fix them in the pre-registration)."""

    zero_percentile: float = 0.0  # -z: percentile of non-zero measurements counted as zero
    exclude_zero_pairs: bool = False  # -x
    model_zero: bool = False  # -y: calculated values at or below the zero level set to 0
    plume_only: bool = False  # -p
    averaging: str | None = None  # -at -> "time", -as -> "space", None: unaveraged
    bootstrap: bool = False  # -b
    seed: int | None = None  # seed for -b (statmain has none)
    contingency_level: float = 1.0  # -l
    concentration_factor: float = 1.0  # -c

    def __post_init__(self):
        if self.averaging not in (None, "time", "space"):
            raise ValueError("averaging must be None, 'time' or 'space'")
        if self.bootstrap and self.seed is None:
            raise ValueError("bootstrap needs an explicit seed")


@dataclass
class DatemStatistics:
    """Every value ``statmain`` prints, under its own labels."""

    n_records: int
    zero_percentile: float
    zero_value: float
    correlation: float
    correlation_significant: bool
    t_value: float
    mean_measured: float
    mean_calculated: float
    ratio: float
    nmse: float
    n_pairs: int
    bias: float
    bias_lo99: float
    bias_hi99: float
    fractional_bias: float
    far: float
    pod: float
    ts: float
    fms: float
    foex: float
    fa2: float
    fa5: float
    fap: float
    fzc: float
    fzm: float
    meas_pct: dict = field(default_factory=dict)
    calc_pct: dict = field(default_factory=dict)
    ksp: float = 0.0
    rank: float = 0.0
    options: dict = field(default_factory=dict)

    @property
    def rank_terms(self) -> dict:
        return dict(
            r2=self.correlation**2,
            fb=1.0 - abs(self.fractional_bias / 2.0),
            fms=self.fms / 100.0,
            ks=1.0 - self.ksp / 100.0,
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({f"{k}_term": v for k, v in self.rank_terms.items()})
        return d


# ---------------------------------------------------------------------------
# pairing (DLOAD) and averaging (AVRGT, AVRGS)
# ---------------------------------------------------------------------------


def pair_predictions(meas: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Attach predictions to measurements as ``statmain``'s DLOAD does.

    A prediction is matched on start date, hour-minute, duration and site
    number; an unmatched measurement gets ``calc = -1`` (missing). When
    several prediction records match, the last one wins. Returns ``meas``
    with columns ``meas`` and ``calc`` in file units.
    """
    out = meas.copy()
    p = pred.drop_duplicates(KEYS, keep="last")[KEYS + ["conc_raw"]]
    merged = out[KEYS].merge(p, on=KEYS, how="left")
    out["meas"] = out["conc_raw"].to_numpy(dtype=float).astype(F)
    out["calc"] = merged["conc_raw"].fillna(-1.0).to_numpy(dtype=float).astype(F)
    return out


def _average(
    keys: list[tuple], meas: np.ndarray, calc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list]:
    order: dict = {}
    for k in keys:
        order.setdefault(k, len(order))
    idx = np.array([order[k] for k in keys])
    ok = (meas >= 0) & (calc >= 0)
    n = np.zeros(len(order), F)
    sm = np.zeros(len(order), F)
    sc = np.zeros(len(order), F)
    np.add.at(n, idx[ok], F(1.0))  # sequential, in record order
    np.add.at(sm, idx[ok], meas[ok].astype(F))
    np.add.at(sc, idx[ok], calc[ok].astype(F))
    keep = n > 0
    groups = [k for k, i in order.items() if keep[i]]
    return sm[keep] / n[keep], sc[keep] / n[keep], groups


def average_pairs(paired: pd.DataFrame, averaging: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the (meas, calc) arrays ``statmain`` analyses, padded with -1 to
    the original record count after averaging, as ``statmain`` leaves them."""
    meas = paired["meas"].to_numpy().astype(F)
    calc = paired["calc"].to_numpy().astype(F)
    if averaging is None:
        return meas, calc
    if averaging == "time":
        keys = list(paired["site_num"])
    else:
        # statmain's own serial time, with the hour field in hhmm.
        obtime = (paired["yr"] * 365 + paired["mo"] * 31 + paired["dy"]) * 24 + paired["hhmm"]
        keys = list(zip(obtime, paired["dur_hhmm"], strict=True))
    m, c, _ = _average(keys, meas, calc)
    pad = len(meas) - len(m)
    return np.concatenate([m, -np.ones(pad, F)]), np.concatenate([c, -np.ones(pad, F)])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def percentiles99(values: np.ndarray) -> np.ndarray:
    """statmain's SORT: values at 1..99 percent, index 0 unused (NaN)."""
    v = np.sort(np.asarray(values, dtype=F), kind="stable")
    out = np.full(100, np.nan)
    n = len(v)
    if n <= 1:  # statmain leaves the array unset and prints a warning
        return out
    i = np.arange(1, 100)
    j = np.maximum(1, (np.float32(i * n) / np.float32(100.0)).astype(np.int64))
    out[1:] = v[j - 1]
    return out


def _zero_level(meas: np.ndarray, pct: float) -> float:
    if pct == 0.0:
        return 0.0
    pct = max(0.0, min(99.0, pct))
    return float(percentiles99(meas[meas > 0])[int(pct)])


def _select(meas, calc, zero, opt: StatOptions):
    """Apply the missing, zero and pair-selection rules shared by every routine."""
    ok = (meas >= 0) & (calc >= 0)
    mm = np.where(meas <= zero, 0.0, meas)
    cc = np.where(opt.model_zero & (calc <= zero), 0.0, calc)
    pick = np.zeros_like(ok)
    plume = opt.plume_only
    excl = opt.exclude_zero_pairs and not plume  # statmain: -p overrides -x
    if plume:
        pick |= (mm > 0) & (cc > 0)
    if excl:
        pick |= (mm > 0) | (cc > 0)
    if not excl and not plume:
        pick[:] = True
    sel = ok & pick
    return mm[sel], cc[sel]


def _nint(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


_T_LINEAR = [6.96, 3.75, 2.90, 2.58, 2.46, 2.39, 2.36, 2.33]
_T_BIAS = [9.92, 4.60, 3.36, 2.92, 2.75, 2.66, 2.62, 2.58]


def _linear(mm: np.ndarray, cc: np.ndarray) -> dict:
    k = F(len(mm))
    if k <= 2:
        return dict(mbar=0.0, cbar=0.0, n=0, corl=0.0, nmse=0.0, sig=False, teqv=0.0)
    sm, sc = _ssum(mm), _ssum(cc)
    sqm, sqc, sqb, sqd = _ssum(mm * mm), _ssum(cc * cc), _ssum(mm * cc), _ssum((cc - mm) ** 2)
    mbar, cbar = sm / k, sc / k
    nmse = F(999.99) if (mbar == 0 or cbar == 0) else sqd / mbar / cbar / k
    syy = sqc - sc * sc / k
    sxx = sqm - sm * sm / k
    sxy = sqb - sc * sm / k
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        var = syy - sxy * sxy / sxx
        slope = sxy / sxx
        corl = F(0.0) if (sxx == 0 or syy == 0) else sxy / np.sqrt(sxx * syy)
        ser = np.sqrt(var / sxx / (k - F(2.0)))
    teqv = F(999.99) if (ser == 0 or not np.isfinite(ser)) else abs(slope) / ser
    kk = max(1, min(8, _nint(math.log(float(k) - 2.0) / math.log(2.0))))
    return dict(
        mbar=float(mbar),
        cbar=float(cbar),
        n=int(k),
        corl=float(corl),
        nmse=float(nmse),
        sig=bool(teqv > F(_T_LINEAR[kk - 1])),
        teqv=float(teqv),
    )


def _bootstrap_r2(meas, calc, zero, opt: StatOptions) -> float:
    """Mean of r^2 over N passes, each dropping every record with p = 0.2."""
    rng = np.random.default_rng(opt.seed)
    n_rec = len(meas)
    # Per-record contribution, zero where the record is not selected.
    ok = (meas >= 0) & (calc >= 0)
    mm = np.where(meas <= zero, 0.0, meas)
    cc = np.where(opt.model_zero & (calc <= zero), 0.0, calc)
    plume = opt.plume_only
    excl = opt.exclude_zero_pairs and not plume
    if plume:
        pick = (mm > 0) & (cc > 0)
    elif excl:
        pick = (mm > 0) | (cc > 0)
    else:
        pick = np.ones(n_rec, bool)
    w0 = (ok & pick).astype(float)
    mm = np.where(w0 > 0, mm, 0.0)
    cc = np.where(w0 > 0, cc, 0.0)
    cols = np.stack([w0, mm, cc, mm * mm, cc * cc, mm * cc], axis=1)
    rsum = 0.0
    chunk = max(1, int(2.0e7 // max(n_rec, 1)))
    done = 0
    while done < n_rec:
        m = min(chunk, n_rec - done)
        keep = rng.random((m, n_rec)) >= 0.2
        s = keep.astype(float) @ cols
        k, sm, sc, sqm, sqc, sqb = s.T
        with np.errstate(divide="ignore", invalid="ignore"):
            syy = sqc - sc * sc / k
            sxx = sqm - sm * sm / k
            sxy = sqb - sc * sm / k
            r = np.where((sxx == 0) | (syy == 0), 0.0, sxy / np.sqrt(sxx * syy))
        r = np.where(k > 2, r, 0.0)
        rsum += float((r * r).sum())
        done += m
    return rsum / n_rec


def datem_statistics(
    meas: np.ndarray, calc: np.ndarray, options: StatOptions | None = None
) -> DatemStatistics:
    """Compute ``statmain``'s statistics on already paired and averaged arrays
    (negative = missing), as :func:`score` does after pairing."""
    opt = options or StatOptions()
    meas = np.asarray(meas, dtype=F) * F(opt.concentration_factor)
    calc = np.asarray(calc, dtype=F) * F(opt.concentration_factor)
    n_records = int(((meas >= 0) | (calc >= 0)).sum()) if opt.averaging else len(meas)
    zero = _zero_level(meas, opt.zero_percentile)
    mm, cc = _select(meas, calc, zero, opt)

    lin = _linear(mm, cc)
    corl = lin["corl"]
    if opt.bootstrap:
        corl = math.sqrt(_bootstrap_r2(meas, calc, zero, opt))

    # BIASES
    k = F(len(mm))
    bias = _ssum(cc - mm) / k if k > 0 else F(0.0)
    if k > 0:
        kk = max(1, min(8, _nint(math.log(float(k)) / math.log(2.0))))
        with np.errstate(divide="ignore", invalid="ignore"):
            cint = F(_T_BIAS[kk - 1]) * np.sqrt(_ssum((cc - mm - bias) ** 2) / (k * (k - F(1.0))))
    else:
        cint = F(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = F(2.0) * bias / (F(lin["cbar"]) + F(lin["mbar"]))
        ratio = F(lin["cbar"]) / F(lin["mbar"]) if lin["mbar"] else float("nan")

    # SPATIAL: raw arrays, missing included, as statmain does.
    lvl = opt.contingency_level
    hit = int(((meas >= lvl) & (calc >= lvl)).sum())
    miss = int(((meas >= lvl) & (calc < lvl)).sum())
    false = int(((meas < lvl) & (calc >= lvl)).sum())
    far = 0.0 if false + hit == 0 else 100.0 * false / (false + hit)
    ts = 100.0 * hit / (false + hit + miss) if (false + hit + miss) else float("nan")
    pod = 100.0 * hit / (hit + miss) if (hit + miss) else float("nan")

    # FIGMST
    union = int(((mm > 0) | (cc > 0)).sum())
    inter = int(((mm > 0) & (cc > 0)).sum())
    fms = float(F(100.0) * F(inter) / F(union)) if union else 0.0

    # SCATER
    nb = len(mm)
    numx = int((cc > mm).sum())
    nzm = int((mm == 0).sum())
    nzc = int(((mm != 0) & (cc == 0)).sum())
    both = (mm != 0) & (cc != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_mc = np.where(both, cc / np.where(mm == 0, F(1), mm), F(np.nan)).astype(F)
    num2 = int((both & (ratio_mc >= F(0.5)) & (ratio_mc <= F(2.0))).sum())
    num5 = int((both & (ratio_mc >= F(0.2)) & (ratio_mc <= F(5.0))).sum())
    nump = int(both.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        foex = 100.0 * numx / nb - 50.0 if nb else float("nan")
        fa2, fa5, fap = (100.0 * x / nb if nb else float("nan") for x in (num2, num5, nump))
        fzc, fzm = (100.0 * x / nb if nb else float("nan") for x in (nzc, nzm))

    # CUMDIST and KSPARM
    msort, csort = percentiles99(mm), percentiles99(cc)
    ksp = 0.0
    for km in range(5, 96):
        if msort[km] != msort[km - 1]:
            for kc in range(5, 96):
                if csort[kc] == csort[kc - 1]:
                    continue
                if csort[kc] >= msort[km]:
                    ksp = max(ksp, abs(float(kc - km)))
                    break

    corl = F(corl)
    rank = (
        corl * corl
        + (F(1.0) - abs(frac / F(2.0)))
        + F(fms) / F(100.0)
        + (F(1.0) - F(ksp) / F(100.0))
    )
    corl, frac, rank, bias, cint = (float(x) for x in (corl, frac, rank, bias, cint))
    pcts = (95, 90, 75, 50)
    return DatemStatistics(
        n_records=n_records,
        zero_percentile=opt.zero_percentile,
        zero_value=zero,
        correlation=corl,
        correlation_significant=bool(lin["sig"]),
        t_value=lin["teqv"],
        mean_measured=lin["mbar"],
        mean_calculated=lin["cbar"],
        ratio=ratio,
        nmse=lin["nmse"],
        n_pairs=lin["n"],
        bias=bias,
        bias_lo99=bias - cint,
        bias_hi99=bias + cint,
        fractional_bias=frac,
        far=far,
        pod=pod,
        ts=ts,
        fms=fms,
        foex=foex,
        fa2=fa2,
        fa5=fa5,
        fap=fap,
        fzc=fzc,
        fzm=fzm,
        meas_pct={p: float(msort[p]) for p in pcts},
        calc_pct={p: float(csort[p]) for p in pcts},
        ksp=ksp,
        rank=rank,
        options=asdict(opt),
    )


def score(meas: pd.DataFrame, pred: pd.DataFrame, options: StatOptions | None = None):
    """Pair DATEM-format measurements and predictions and score them."""
    opt = options or StatOptions()
    paired = pair_predictions(meas, pred)
    m, c = average_pairs(paired, opt.averaging)
    return datem_statistics(m, c, opt)


# ---------------------------------------------------------------------------
# statmain's printed file
# ---------------------------------------------------------------------------

#: (label in statA/statT files, attribute or callable)
_FIELDS = [
    ("Percentile input for zero measured", "zero_percentile"),
    ("Zero measured concentration value", "zero_value"),
    ("Correlation coefficient", "correlation"),
    ("T-value (|Slope|/Standard Error)", "t_value"),
    ("Average measured concentration", "mean_measured"),
    ("Average calculated concentration", "mean_calculated"),
    ("Ratio of calculated/measured", "ratio"),
    ("Normalized mean square error", "nmse"),
    ("Number of pairs analyzed", "n_pairs"),
    ("Average bias [(C-M)/N]", "bias"),
    ("Lo 99 % confidence interval", "bias_lo99"),
    ("Hi 99 % confidence interval", "bias_hi99"),
    ("Fractional bias [2B/(C+M)]", "fractional_bias"),
    ("False Alarm Rate [fa/(fa+hit)]", "far"),
    ("Probability of Detection [hit/(hit+miss)]", "pod"),
    ("Threat Score [hit/(fa+hit+miss)]", "ts"),
    ("Fig of merit in space (%)", "fms"),
    ("Factor exceeding [N(C>M)/N-0.5]", "foex"),
    ("Percent C/M ~ 2", "fa2"),
    ("Percent C/M ~ 5", "fa5"),
    ("Percent M>0 and C>0", "fap"),
    ("Percent M>0 and C=0", "fzc"),
    ("Percent M=0 and C>0", "fzm"),
    ("Measured   95-th percentile", ("meas_pct", 95)),
    ("Measured   90-th percentile", ("meas_pct", 90)),
    ("Measured   75-th percentile", ("meas_pct", 75)),
    ("Measured   50-th percentile", ("meas_pct", 50)),
    ("Calculated 95-th percentile", ("calc_pct", 95)),
    ("Calculated 90-th percentile", ("calc_pct", 90)),
    ("Calculated 75-th percentile", ("calc_pct", 75)),
    ("Calculated 50-th percentile", ("calc_pct", 50)),
    ("Kolmogorov-Smirnov Parameter", "ksp"),
    ("Final rank (C,FB,FMS,KSP)", "rank"),
]


def _value(stats: DatemStatistics, key):
    if isinstance(key, tuple):
        return getattr(stats, key[0])[key[1]]
    return getattr(stats, key)


def parse_stat_file(path) -> list[tuple[str, str]]:
    """Read a ``statA``/``statT`` file as (label, printed value) in order,
    skipping the header and the record-count line."""
    out = []
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    for line in lines[5:]:
        if not line.strip():
            continue
        val, label = line[:10], line[10:].strip()
        if "data points" in label.lower():
            out.append(("records", val.strip()))
            continue
        out.append((label, val.strip()))
    return out


def format_value(v, integer: bool = False) -> str:
    if integer:
        return f"{int(v):10d}".strip()
    return f"{v:10.2f}".strip()


def compare_with_stat_file(stats: DatemStatistics, path) -> list[tuple[str, str, str]]:
    """Return (label, ours, statmain's) for every printed value that differs."""
    printed = parse_stat_file(path)
    ours = [("records", format_value(stats.n_records, True))]
    for label, key in _FIELDS:
        ours.append((label, format_value(_value(stats, key), key == "n_pairs")))
    if len(printed) != len(ours):
        raise ValueError(f"{path}: {len(printed)} values, expected {len(ours)}")
    diffs = []
    for (_lab_p, val_p), (lab_o, val_o) in zip(printed, ours, strict=True):
        if val_p != val_o:
            diffs.append((lab_o, val_o, val_p))
    return diffs
