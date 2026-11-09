import bisect
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(os.path.dirname(HERE), "data", "macro", "fred_20261108")
MACRO_F = ("r10", "d_r10_20", "d_r10_60", "d_r2_60", "curve", "d_ff_120", "vix", "d_vix_20", "fx20")
MACRO_LABEL = {"r10": "미국 10년 금리", "d_r10_20": "10년 금리 20일 변화", "d_r10_60": "10년 금리 60일 변화", "d_r2_60": "2년 금리 60일 변화", "curve": "장단기 금리차(10년-2년)", "d_ff_120": "기준금리 6개월 변화", "vix": "공포지수(VIX)", "d_vix_20": "VIX 20일 변화", "fx20": "원달러 20일 변화"}


def read_series(name, d=DIR):
    days, vals = [], []
    with open(os.path.join(d, name + ".csv"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0][:1] not in "12":
                continue
            try:
                v = float(row[1])
            except ValueError:
                continue
            days.append(row[0])
            vals.append(v)
    return days, np.array(vals)


class Macro:
    def __init__(self, fx_days, fx_vals, d=DIR):
        self.s = {k: read_series(k, d) for k in ("DGS10", "DGS2", "T10Y2Y", "DFF", "VIXCLS")}
        self.fx = (list(fx_days), np.asarray(fx_vals, float))

    @staticmethod
    def back(series, D, lag):
        days, v = series
        i = bisect.bisect_left(days, D) - 1
        j = i - lag
        if i < 0 or j < 0:
            return None, None
        return v[i], v[j]

    def at(self, D):
        g = lambda k, lag: self.back(self.s[k], D, lag)
        r10, r10_20 = g("DGS10", 20)
        _, r10_60 = g("DGS10", 60)
        r2, r2_60 = g("DGS2", 60)
        cur, _ = g("T10Y2Y", 0)
        ff, ff_120 = g("DFF", 175)
        vix, vix_20 = g("VIXCLS", 20)
        fx, fx_20 = self.back(self.fx, D, 20)
        vals = [r10, None if r10_20 is None else r10 - r10_20, None if r10_60 is None else r10 - r10_60, None if r2_60 is None else r2 - r2_60, cur, None if ff_120 is None else ff - ff_120, None if vix is None else math.log(vix), None if vix_20 is None else math.log(vix / vix_20), None if fx_20 is None else math.log(fx / fx_20)]
        if any(x is None for x in vals):
            return None
        return np.array(vals, float)


DIFF_F = ("kr_us10", "kr_us3m", "jp_us10", "de_us10", "cn_us3m", "d_kr_us10_3m", "d_kr_us10_20")
DIFF_LABEL = {"kr_us10": "한국-미국 장기금리 차", "kr_us3m": "한국-미국 단기금리 차", "jp_us10": "일본-미국 장기금리 차", "de_us10": "독일-미국 장기금리 차", "cn_us3m": "중국-미국 단기금리 차", "d_kr_us10_3m": "한미 장기금리 차 3개월 변화", "d_kr_us10_20": "한미 장기금리 차 20일 변화(국고채 ETF 추정)"}


def monthly(name, d=DIR, lag=2):
    days, v = read_series(name, d)
    avail = []
    for x in days:
        y, mo = int(x[:4]), int(x[5:7]) + lag
        y += (mo - 1) // 12
        mo = (mo - 1) % 12 + 1
        avail.append(f"{y:04d}-{mo:02d}-01")
    return avail, v


def known(series, D, back=0):
    days, v = series
    i = bisect.bisect_right(days, D) - 1 - back
    return v[i] if i >= 0 else None


class RateDiff:
    def __init__(self, kr10_days, kr10_close, d=DIR, duration=7.5):
        self.m = {k: monthly(k, d) for k in ("IRLTLT01KRM156N", "IR3TIB01KRM156N", "IRLTLT01JPM156N", "IRLTLT01DEM156N", "IR3TIB01CNM156N", "IRLTLT01USM156N", "IR3TIB01USM156N")}
        self.us10 = read_series("DGS10", d)
        self.kr_days = list(kr10_days)
        self.kr_lc = np.log(np.asarray(kr10_close, float))
        self.dur = duration

    def at(self, D):
        g = lambda k, b=0: known(self.m[k], D, b)
        kr10, us10, kr3, us3 = g("IRLTLT01KRM156N"), g("IRLTLT01USM156N"), g("IR3TIB01KRM156N"), g("IR3TIB01USM156N")
        jp10, de10, cn3 = g("IRLTLT01JPM156N"), g("IRLTLT01DEM156N"), g("IR3TIB01CNM156N")
        kr10b, us10b = g("IRLTLT01KRM156N", 3), g("IRLTLT01USM156N", 3)
        i = bisect.bisect_left(self.kr_days, D) - 1
        u_now, u_20 = Macro.back(self.us10, D, 20)
        if None in (kr10, us10, kr3, us3, jp10, de10, cn3, kr10b, us10b, u_now, u_20):
            return None
        dkr = -(self.kr_lc[i] - self.kr_lc[i - 20]) * 100 / self.dur if i >= 20 else (u_now - u_20)
        return np.array([kr10 - us10, kr3 - us3, jp10 - us10, de10 - us10, cn3 - us3, (kr10 - us10) - (kr10b - us10b), dkr - (u_now - u_20)], float)
