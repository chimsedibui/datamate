"""Restore pandas 1.x/2.x APIs that pandas 3 removed, for LLM-written code (loaders and solve()).

Qwen3-14B writes pandas from its training data: `fillna(method="ffill")`, `df.append`, `applymap`, `iteritems`,
`resample("M")`, `.dt.week`, `pd.value_counts`. Under pandas 3 each of these raises, and a 14B model often repeats
the same call on retry (a public-test build failed 4 times on `fillna(method=...)`). The shims map each old call to
its current equivalent with the same result. Installed once when the package is imported; idempotent.
"""
from __future__ import annotations

import functools
import re

import pandas as pd

_FREQ = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE", "BM": "BME", "BQ": "BQE", "BA": "BYE", "BY": "BYE",
         "SM": "SME", "H": "h", "BH": "bh", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns"}
_FREQ_RX = re.compile(r"^(\d*)(BM|BQ|BA|BY|SM|BH|M|Q|Y|A|H|T|S|L|U|N)(-[A-Za-z]+)?$")


def _freq(f):
    """Old offset alias -> current one ('M' -> 'ME', '2H' -> '2h'); anything else unchanged."""
    if not isinstance(f, str):
        return f
    m = _FREQ_RX.match(f.strip())
    return f"{m.group(1)}{_FREQ[m.group(2)]}{m.group(3) or ''}" if m else f


def _wrap_freq(owner, name, pos: int | None):
    orig = getattr(owner, name)
    if getattr(orig, "_dm_compat", False):
        return

    @functools.wraps(orig)
    def f(*args, **kw):
        for k in ("freq", "rule"):
            if k in kw:
                kw[k] = _freq(kw[k])
        if pos is not None and len(args) > pos:
            args = (*args[:pos], _freq(args[pos]), *args[pos + 1:])
        return orig(*args, **kw)
    f._dm_compat = True
    setattr(owner, name, f)


def _fillna(orig):
    @functools.wraps(orig)
    def f(self, value=None, *args, method=None, **kw):
        if method is not None and value is None:
            limit = kw.get("limit")
            m = str(method).lower()
            if m in ("ffill", "pad"):
                return self.ffill(limit=limit) if not kw.get("inplace") else self._update_inplace(self.ffill(limit=limit))
            if m in ("bfill", "backfill"):
                return self.bfill(limit=limit) if not kw.get("inplace") else self._update_inplace(self.bfill(limit=limit))
        return orig(self, value, *args, **kw)
    f._dm_compat = True
    return f


def install() -> None:
    for cls in (pd.DataFrame, pd.Series):
        if not getattr(cls.fillna, "_dm_compat", False):
            cls.fillna = _fillna(cls.fillna)
        if not hasattr(cls, "append"):
            cls.append = lambda self, other, ignore_index=False, **_: pd.concat(
                [self, *(other if isinstance(other, (list, tuple)) else [other])], ignore_index=ignore_index)
        if not hasattr(cls, "iteritems"):
            cls.iteritems = cls.items
        if not hasattr(cls, "bool"):
            cls.bool = lambda self: bool(self.squeeze())
        if not hasattr(cls, "swapaxes"):
            cls.swapaxes = lambda self, *a, **k: self.T
        _wrap_freq(cls, "resample", 1)
        _wrap_freq(cls, "asfreq", 1)
    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = pd.DataFrame.map
    if not hasattr(pd, "value_counts"):
        pd.value_counts = lambda values, **k: pd.Series(values).value_counts(**k)
    _wrap_freq(pd, "date_range", 3)
    _wrap_freq(pd, "bdate_range", 3)
    from pandas.core.resample import TimeGrouper  # pd.Grouper(freq=...) is constructed as a TimeGrouper
    if not getattr(TimeGrouper.__init__, "_dm_compat", False):
        orig_init = TimeGrouper.__init__

        @functools.wraps(orig_init)
        def ginit(self, *args, **kw):
            if "freq" in kw:
                kw["freq"] = _freq(kw["freq"])
            orig_init(self, *args, **kw)
        ginit._dm_compat = True
        TimeGrouper.__init__ = ginit
    from pandas.core.indexes.accessors import DatetimeProperties
    if not hasattr(DatetimeProperties, "week"):
        DatetimeProperties.week = property(lambda self: self.isocalendar().week.astype("int64"))
        DatetimeProperties.weekofyear = DatetimeProperties.week
