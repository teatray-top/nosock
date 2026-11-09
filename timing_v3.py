import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction

import numpy as np

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\timing_actionvalue_v1_spec_20261106.json")
SPEC_SHA256 = "c9c0087744b19083043b7686f4c3f8148cb6fa1814b3913c2c8101df3763e98f"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\timing_actionvalue_v1_20261106")
SCHEMA = "timing-actionvalue-v1"
MODULE = "timing_actionvalue_v1"
SYM = "069500"
CUTOFF = "2026-11-05"
FULL_ROWS, RETAINED_ROWS = 5939, 5938
H = 20
FEATURE_START = 199
FIRST_EXEC_YEAR, LAST_EXEC_YEAR = 2015, 2026
ALPHA = 1.0
DEPOSIT = 10000
TICK = 5
FEE_NUM, FEE_DEN = 15, 100000
FEATURES = ("logret5", "logret20", "logret60", "ma200_distance", "sd20", "held")
PARENT_ORDER = ("raw_prices", "adjusted_prices", "allocation_mechanics_spec", "module_contract", "timing_spec")
LEDGER_CALLS = [0]
FIT_CALLS = [0]


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    if s["version"] != SCHEMA or s["output"]["directory"] != OUT_DIR or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    av, md = s["action_values"], s["model"]
    if av["N"] != H or av["train_first_decision_index"] != FEATURE_START or md["alpha"] != ALPHA or list(s["features"]["order"]) != list(FEATURES) or s["data"]["cutoff_session"] != CUTOFF or s["data"]["full_input_rows"] != FULL_ROWS or s["data"]["retained_rows"] != RETAINED_ROWS:
        raise InvalidInput("spec design constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices", "allocation_mechanics_spec", "module_contract")}
    parents["timing_spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, folds=s["walk_forward"]["folds"], counts=s["counts"], period=s["ledger"]["period"], external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        if not os.path.exists(p["path"]):
            raise InvalidInput(f"parent {name} missing")
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def fee(x):
    if not isinstance(x, int) or x < 0:
        raise InvalidInput("fee argument must be a non-negative integer")
    return x * FEE_NUM // FEE_DEN


def buy_cost(P):
    return P + TICK + fee(P + TICK)


def sell_proceeds(P):
    return P - TICK - fee(P - TICK)


def targets(O, i, h=H):
    P, Q = O[i + 1], O[i + h + 1]
    return (sell_proceeds(Q) - buy_cost(P)) / P, (sell_proceeds(Q) - sell_proceeds(P)) / P


def validate_series(days, O, C_raw, C_adj):
    if not (len(days) == len(O) == len(C_raw) == len(C_adj)):
        raise InvalidInput("array length mismatch")
    if days != sorted(days) or len(set(days)) != len(days):
        raise InvalidInput("sessions not sorted/unique")
    Oi, Ci, Ca = [], [], []
    for o, c, a in zip(O, C_raw, C_adj):
        fo, fc, fa = float(o), float(c), float(a)
        if not (math.isfinite(fo) and math.isfinite(fc) and math.isfinite(fa) and fo.is_integer() and fc.is_integer() and fo > TICK and fc > 0 and fa > 0):
            raise InvalidInput("invalid price (raw integer won with O > tick, C > 0; adjusted positive finite)")
        Oi.append(int(fo))
        Ci.append(int(fc))
        Ca.append(fa)
    return Oi, Ci, Ca


def prepare_prices(raw, adj):
    if not (len(raw["days"]) == len(raw["O"]) == len(raw["C"])) or not (len(adj["days"]) == len(adj["C"])):
        raise InvalidInput("full parallel arrays misaligned")
    if raw["days"] != sorted(raw["days"]) or len(set(raw["days"])) != len(raw["days"]) or adj["days"] != sorted(adj["days"]) or len(set(adj["days"])) != len(adj["days"]):
        raise InvalidInput("full date arrays not sorted/unique")
    if len(raw["days"]) != FULL_ROWS or len(adj["days"]) != FULL_ROWS:
        raise InvalidInput("full input row count differs from the fixed facts")
    if raw["days"] != adj["days"]:
        raise InvalidInput("raw and adjusted full date arrays differ (checked before slicing)")
    keep_r = [k for k, d in enumerate(raw["days"]) if d <= CUTOFF]
    keep_a = [k for k, d in enumerate(adj["days"]) if d <= CUTOFF]
    rd = [raw["days"][k] for k in keep_r]
    ad = [adj["days"][k] for k in keep_a]
    if rd != ad:
        raise InvalidInput("raw and adjusted retained dates differ")
    if len(rd) != RETAINED_ROWS or rd[-1] != CUTOFF or rd[0] != "2002-10-14":
        raise InvalidInput("retained calendar differs from the fixed facts")
    O, Cr, Ca = validate_series(rd, [raw["O"][k] for k in keep_r], [raw["C"][k] for k in keep_r], [adj["C"][k] for k in keep_a])
    return rd, O, Cr, Ca


def load_prices(spec):
    raw = json.load(open(spec["parents"]["raw_prices"]["path"], encoding="utf-8"))[SYM]
    adj = json.load(open(spec["parents"]["adjusted_prices"]["path"], encoding="utf-8"))[SYM]
    return prepare_prices(raw, adj)


def features(C_adj, i):
    if i < FEATURE_START:
        return None
    c = C_adj
    r20 = [math.log(c[t] / c[t - 1]) for t in range(i - 19, i + 1)]
    m = sum(r20) / 20.0
    sd = math.sqrt(sum((x - m) ** 2 for x in r20) / 20.0)
    ma200 = sum(c[i - 199:i + 1]) / 200.0
    f = [math.log(c[i] / c[i - 5]), math.log(c[i] / c[i - 20]), math.log(c[i] / c[i - 60]), c[i] / ma200 - 1.0, sd]
    if not all(math.isfinite(v) for v in f):
        raise InvalidInput("non-finite feature")
    return f


def year_folds(days, first_year=FIRST_EXEC_YEAR, last_year=LAST_EXEC_YEAR, h=H, start=FEATURE_START):
    n = len(days)
    out = []
    firsts = {}
    for y in range(first_year, last_year + 2):
        e = next((k for k, d in enumerate(days) if d >= f"{y}-01-01"), None)
        firsts[y] = e
    for y in range(first_year, last_year + 1):
        e = firsts[y]
        if e is None or e == 0:
            raise InvalidInput(f"no execution session for year {y}")
        c = e - 1
        nxt = firsts.get(y + 1)
        last_dec = (nxt - 2) if nxt is not None else n - 2
        tl = c - h - 1
        out.append(dict(entry_year=y, fit_cutoff_index=c, fit_cutoff_session=days[c], first_entry_session=days[e], last_entry_session=days[last_dec + 1], first_decision_session=days[c], last_decision_session=days[last_dec], first_decision_index=c, last_decision_index=last_dec, decision_count=last_dec - c + 1, train_decisions=tl - start + 1, train_rows=2 * (tl - start + 1), train_first_decision_index=start, train_last_decision_index=tl, train_max_label_available_session=days[tl + h + 1]))
    return out


def training_rows(O, C_adj, cutoff_index, start=FEATURE_START, h=H):
    X, y, used = [], [], []
    for j in range(start, cutoff_index - h):
        if j + h + 1 > cutoff_index:
            raise InvalidInput("unmatured label in training set")
        f = features(C_adj, j)
        t0, t1 = targets(O, j, h)
        if not (math.isfinite(t0) and math.isfinite(t1)):
            raise InvalidInput("non-finite target")
        X.append(f + [0.0])
        y.append(t0)
        X.append(f + [1.0])
        y.append(t1)
        used.append(j)
    return X, y, used


def fit_ridge(X, y, alpha=ALPHA):
    FIT_CALLS[0] += 1
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[0] < 2:
        raise InvalidInput("design matrix shape invalid")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise InvalidInput("non-finite training data")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])
    pen = np.eye(Z.shape[1]) * alpha
    pen[0, 0] = 0.0
    A = Z.T @ Z + pen
    beta = np.linalg.solve(A, Z.T @ y)
    if not np.isfinite(beta).all():
        raise InvalidInput("non-finite coefficients")
    resid = y - Z @ beta
    return dict(beta=[float(b) for b in beta], mu=[float(v) for v in mu], sd=[float(v) for v in sd], n_rows=int(len(y)), condition_number=float(np.linalg.cond(A)), residual_variance_ratio=float(resid.var() / y.var()) if y.var() > 0 else None, sum_squared_residuals=float(resid @ resid), alpha=alpha, finite=True)


def predict(model, feats, held):
    z = [(f - m) / s for f, m, s in zip(list(feats) + [float(held)], model["mu"], model["sd"])]
    b = model["beta"]
    val = b[0] + sum(bi * zi for bi, zi in zip(b[1:], z))
    if not math.isfinite(val):
        raise InvalidInput("non-finite prediction")
    return val


def ledger_run(days, O, C_raw, start, end, decide):
    LEDGER_CALLS[0] += 1
    cash, qty, deposits = 0, 0, 0
    pending = None
    daily, execs, trades, decisions = [], [], [], []
    buys = sells = entries = exits = maintenance = 0
    fees = tick_cost = turnover = 0
    for t in range(start, end + 1):
        executed_w = None
        if pending is not None:
            w, dec_i = pending
            o = O[t]
            W_open = cash + qty * o
            target = math.floor(w * W_open / o) if w > 0 else 0
            cash_b, qty_b = cash, qty
            side, n, px, f = None, 0, None, 0
            if target < qty:
                n, px = qty - target, o - TICK
                f = fee(n * px)
                cash += n * px - f
                qty -= n
                sells += 1
                side = "sell"
                if qty == 0:
                    exits += 1
            elif target > qty:
                px = o + TICK
                q = min(target - qty, cash // px)
                while q > 0 and q * px + fee(q * px) > cash:
                    q -= 1
                if q >= 1:
                    f = fee(q * px)
                    cash -= q * px + f
                    qty += q
                    buys += 1
                    side, n = "buy", q
                    if qty_b == 0:
                        entries += 1
                    else:
                        maintenance += 1
            if side:
                fees += f
                tick_cost += n * TICK
                turnover += n * o
                if (cash + qty * o) - (cash_b + qty_b * o) != -(n * TICK + f) or cash < 0 or qty < 0:
                    raise InvalidInput("ledger invariant violated")
                trades.append(dict(weight_decision_session=days[dec_i], execution_session=days[t], side=side, kind=("entry" if side == "buy" and qty_b == 0 else "maintenance" if side == "buy" else "exit" if qty == 0 else "partial_sell"), qty=n, price=px, fee=f, cash_before=cash_b, cash_after=cash, qty_before=qty_b, qty_after=qty))
            execs.append(dict(weight_decision_session=days[dec_i], execution_session=days[t], observed_open=o, w=w, W_open=W_open, target_qty=target, side=side, realized_qty=n, cash_before=cash_b, cash_after=cash, qty_before=qty_b, qty_after=qty, fee=f, tick=n * TICK))
            executed_w = w
            pending = None
        deposits += DEPOSIT
        cash += DEPOSIT
        value = cash + qty * C_raw[t]
        held = 1 if qty > 0 else 0
        w_next = None
        if t + 1 <= end:
            dec = decide(t, held)
            w_next = dec["w"]
            pending = (w_next, t)
            decisions.append(dict(decision_index=t, decision_session=days[t], entry_session=days[t + 1], held_at_close=held, **dec))
        daily.append(dict(session=days[t], cash=cash, qty=qty, value=value, deposits=deposits, pnl=value - deposits, exposure=(qty * C_raw[t] / value) if value > 0 else 0.0, held=held, executed_w_from_previous_close=executed_w, w_for_next_session=w_next))
    pnl = [d["pnl"] for d in daily]
    peak, dd = 0, 0
    for x in pnl:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    met = dict(start=days[start], end=days[end], sessions=len(daily), unique_daily_rows=len({d["session"] for d in daily}), deposit_events=len(daily), deposits=deposits, pnl_won_after_costs=pnl[-1], pnl_pct_of_deposits=pnl[-1] / deposits if deposits else None, max_dd_won=dd, exposure_avg=sum(d["exposure"] for d in daily) / len(daily), days_held=sum(1 for d in daily if d["held"] == 1), buys=buys, sells=sells, flat_to_long_entries=entries, long_to_flat_exits=exits, long_maintenance_buys=maintenance, trades=len(trades), execution_opportunities=len(execs), fees_total_won=fees, tick_cost_won=tick_cost, turnover_won=turnover, qty_end=qty, final_cash=cash, final_value=daily[-1]["value"])
    return dict(metrics=met, daily=daily, executions=execs, trades=trades, decisions=decisions)


def yearly_increments(daily):
    last = {}
    for d in daily:
        last[d["session"][:4]] = d["pnl"]
    out, prev = {}, 0
    for y in sorted(last):
        out[y] = last[y] - prev
        prev = last[y]
    return out


def yearly_trade_counts(trades):
    out = {}
    for t in trades:
        y = t["execution_session"][:4]
        g = out.setdefault(y, dict(buys=0, sells=0, fees=0, ticks=0))
        g["buys" if t["side"] == "buy" else "sells"] += 1
        g["fees"] += t["fee"]
        g["ticks"] += t["qty"] * TICK
    return out


def holding_streaks(daily):
    runs, cur = [], 0
    for d in daily:
        if d["held"] == 1:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if not runs:
        return dict(count=0)
    s = sorted(runs)
    return dict(count=len(runs), min=s[0], median=s[len(s) // 2], max=s[-1], mean=sum(s) / len(s))


def build_models(days, O, C_adj, folds):
    models = {}
    for fd in folds:
        c = fd["fit_cutoff_index"]
        X, yv, used = training_rows(O, C_adj, c)
        if len(used) != fd["train_decisions"] or len(yv) != fd["train_rows"] or used[0] != fd["train_first_decision_index"] or used[-1] != fd["train_last_decision_index"]:
            raise InvalidInput(f"training rows differ from fold facts for {fd['entry_year']}")
        m = fit_ridge(X, yv)
        m.update(entry_year=fd["entry_year"], fit_cutoff_index=c, fit_cutoff_session=days[c], first_entry_session=fd["first_entry_session"], train_decisions=len(used), train_first_decision_session=days[used[0]], train_last_decision_session=days[used[-1]], train_max_label_available_session=days[used[-1] + H + 1], target_mean=float(np.mean(yv)), target_sd=float(np.std(yv)))
        models[fd["entry_year"]] = m
    return models


def make_decider(days, C_adj, models, folds, log):
    by_dec = {}
    for fd in folds:
        for i in range(fd["first_decision_index"], fd["last_decision_index"] + 1):
            by_dec[i] = fd["entry_year"]

    def decide(i, held):
        y = by_dec.get(i)
        if y is None or y not in models:
            raise InvalidInput(f"no model for decision {days[i]}")
        f = features(C_adj, i)
        pred = predict(models[y], f, held)
        if pred > 0:
            w, action = 1.0, "long"
        elif pred < 0:
            w, action = 0.0, "flat"
        else:
            w, action = float(held), "keep"
        rec = dict(fit_entry_year=y, fit_cutoff_session=models[y]["fit_cutoff_session"], train_max_label_available_session=models[y]["train_max_label_available_session"], features=dict(zip(FEATURES, f + [held])), predicted_contrast=pred, action=action, w=w)
        log.append(rec)
        return rec

    return decide


def compute(days, O, C_raw, C_adj, start, end, folds):
    models = build_models(days, O, C_adj, folds)
    log = []
    LEDGER_CALLS[0] = 0
    M1 = ledger_run(days, O, C_raw, start, end, make_decider(days, C_adj, models, folds, log))
    R1 = ledger_run(days, O, C_raw, start, end, lambda i, held: dict(fit_entry_year=None, fit_cutoff_session=None, train_max_label_available_session=None, features=None, predicted_contrast=None, action="long", w=1.0))
    sign = {}
    for d in M1["decisions"]:
        i = d["decision_index"]
        y = d["entry_session"][:4]
        g = sign.setdefault(y, dict(n=0, matured=0, sign_agree=0, pred_pos=0, pred_neg=0, pred_zero=0, desired_long=0))
        g["n"] += 1
        g["pred_pos"] += int(d["predicted_contrast"] > 0)
        g["pred_neg"] += int(d["predicted_contrast"] < 0)
        g["pred_zero"] += int(d["predicted_contrast"] == 0)
        g["desired_long"] += int(d["w"] == 1.0)
        if i + H + 1 <= len(days) - 1:
            t0, t1 = targets(O, i)
            realized = t1 if d["held_at_close"] == 1 else t0
            g["matured"] += 1
            g["sign_agree"] += int((d["predicted_contrast"] > 0) == (realized > 0))
    for g in sign.values():
        g["sign_agreement_rate_matured"] = g["sign_agree"] / g["matured"] if g["matured"] else None
    inc = dict(M1=yearly_increments(M1["daily"]), R1=yearly_increments(R1["daily"]))
    inc["M1_minus_R1"] = {y: inc["M1"][y] - inc["R1"][y] for y in inc["M1"]}
    diff = [b["pnl"] - a["pnl"] for a, b in zip(R1["daily"], M1["daily"])]
    paired = dict(m1_minus_r1_terminal_pnl=diff[-1], daily_diff_min=min(diff), daily_diff_max=max(diff), sessions_m1_fewer_shares=sum(1 for a, b in zip(R1["daily"], M1["daily"]) if b["qty"] < a["qty"]))
    return dict(models=models, M1=M1, R1=R1, yearly=inc, yearly_trades=dict(M1=yearly_trade_counts(M1["trades"]), R1=yearly_trade_counts(R1["trades"])), sign_diagnostics=sign, holding_streaks=holding_streaks(M1["daily"]), paired=paired, ledger_calls=LEDGER_CALLS[0], fit_calls=FIT_CALLS[0])


def summary_lines(res):
    a, b = res["M1"]["metrics"], res["R1"]["metrics"]
    p = res["paired"]
    lines = [f"[{a['start']}~{a['end']} {a['sessions']}세션 입금 {a['deposits']:,} 적합 {len(res['models'])}회 원장 {res['ledger_calls']}개 결정 {len(res['M1']['decisions'])}]"]
    for tag, m in (("M1 타이밍", a), ("R1 항상 롱", b)):
        lines.append(f"  {tag}: 손익 {m['pnl_won_after_costs']:+,} ({m['pnl_pct_of_deposits']*100:+.1f}%) 낙폭 {m['max_dd_won']:+,} 노출 {m['exposure_avg']*100:.0f}% 보유일 {m['days_held']} 진입 {m['flat_to_long_entries']} 청산 {m['long_to_flat_exits']} 유지매수 {m['long_maintenance_buys']} 매수 {m['buys']} 매도 {m['sells']} 수수료 {m['fees_total_won']:,} 틱 {m['tick_cost_won']:,} 회전 {m['turnover_won']:,} 기말 현금 {m['final_cash']:,} 수량 {m['qty_end']} 가치 {m['final_value']:,}")
    lines.append(f"  M1−R1 기말 {p['m1_minus_r1_terminal_pnl']:+,} | 일별 차이 최소 {p['daily_diff_min']:+,} 최대 {p['daily_diff_max']:+,} | M1 보유주 적은 세션 {p['sessions_m1_fewer_shares']}")
    lines.append("  연도별 증분 M1−R1: " + ", ".join(f"{y} {v:+,}" for y, v in res["yearly"]["M1_minus_R1"].items()))
    lines.append("  연도별 증분 M1: " + ", ".join(f"{y} {v:+,}" for y, v in res["yearly"]["M1"].items()))
    lines.append("  연도별 증분 R1: " + ", ".join(f"{y} {v:+,}" for y, v in res["yearly"]["R1"].items()))
    lines.append("  예측 부호(진입연도): " + ", ".join(f"{y} +{g['pred_pos']}/−{g['pred_neg']}/0:{g['pred_zero']} 일치 {fmt_pct(g['sign_agreement_rate_matured'])} (성숙 {g['matured']})" for y, g in sorted(res["sign_diagnostics"].items())))
    lines.append(f"  보유 연속구간: {res['holding_streaks']}")
    for y, m in res["models"].items():
        lines.append(f"  fit {y}: cutoff {m['fit_cutoff_session']} 행 {m['n_rows']} 조건수 {m['condition_number']:.3e} 잔차분산비 {m['residual_variance_ratio']:.4f} 목표평균 {m['target_mean']:+.3e} β " + ", ".join(f"{v:+.3e}" for v in m["beta"]))
    return lines


def fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "해당 없음"


def build_payload(spec, res, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates (identical to adjusted after the cutoff slice)", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", execution_marker="next_session_open"), symbol=SYM, availability_status="available", decision_at="session_close (signed action-value contrast from closes through i and the actual held state; logical marker)", execution_not_before="next_session_open (quantity decided after observing the open; idealized fill at O +/- 5, no actual order)", model=dict(name="Ridge linear regression", alpha=ALPHA, intercept="unpenalized", objective="sum of squared residuals + 1.0 x sum(coefficients^2) on standardized six columns", solver="numpy float64 linalg.solve on augmented normal equations", features=list(FEATURES), standardization="per fold mean and population sd over the 2n training rows; zero sd -> 1", output="signed action-value contrast (dimensionless); not a probability"), action_values=dict(N=H, entry="P = raw O[i+1]", endpoint="Q = raw O[i+21]", fee="floor(n x 15 / 100000) per integer won notional", buy_cost="b(P) = P + 5 + F(P+5)", net_sell="s(P) = P - 5 - F(P-5)", target_cash="z(i,0) = (s(Q) - b(P)) / P", target_held="z(i,1) = (s(Q) - s(P)) / P", caveat="one-share proxy with no intervening trades; the realized daily policy need not hold 20 sessions; not an optimal dynamic value"), walk_forward=dict(fold_by="calendar year of next-open execution", fits=len(res["models"]), continuous_account="one path 2014-12-30 .. 2026-11-05, no yearly reset; the old position enters the new-year model as the held state", caveat="all historical periods viewed; retrospective walk-forward; no sealed window or prospective validation"), ledger=dict(mechanics="allocate-v1 timeline; integer won; deposit 10,000 at each close; idealized fills; zero tax, zero cash yield, no dividends/distributions", baseline="R1 weight 1 at every close with identical mechanics; cash identity PnL 0"), units=dict(cash="krw (integer)", value="krw (integer)", pnl="krw (integer)", qty="shares (integer)", w="weight in {0,1}", predicted_contrast="dimensionless per-share value contrast", pnl_pct_of_deposits="fraction, not annualized"), counts=dict(real_model_fits=len(res["models"]), fit_calls=res["fit_calls"], total_training_rows_across_fits=sum(m["n_rows"] for m in res["models"].values()), decision_predictions=len(res["M1"]["decisions"]), ledger_paths=res["ledger_calls"], daily_rows_per_ledger={k: res[k]["metrics"]["unique_daily_rows"] for k in ("M1", "R1")}, execution_opportunities_per_ledger={k: res[k]["metrics"]["execution_opportunities"] for k in ("M1", "R1")}, deposit_events_per_ledger={k: res[k]["metrics"]["deposit_events"] for k in ("M1", "R1")}, deposits_won_per_ledger={k: res[k]["metrics"]["deposits"] for k in ("M1", "R1")}, optimizer_calls=0, hyperparameter_searches=0, api_calls=0, orders=0), claims="no guaranteed profit or live advantage; a linear additive-state model may fail to learn timing; failure alone does not diagnose overfitting or unpredictability; the policy is not optimal; no new prospective validation in this run", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], real_experiments=1, gpu=0, api_calls=0, orders=0))
    ref = dict(envelope_artifact_id=artifact_id)
    return dict(envelope=env, lines=summary_lines(res), fits={str(y): m for y, m in res["models"].items()}, decision_records=[dict(d, **ref) for d in res["M1"]["decisions"]], variants={k: dict(metrics=res[k]["metrics"], execution_records=[dict(e, **ref) for e in res[k]["executions"]], trade_records=[dict(t, **ref) for t in res[k]["trades"]], daily_records=[dict(d, **ref) for d in res[k]["daily"]]) for k in ("M1", "R1")}, paired=res["paired"], annual=dict(increments=res["yearly"], trades=res["yearly_trades"]), sign_diagnostics=res["sign_diagnostics"], holding_streaks=res["holding_streaks"], cash_reference=dict(pnl=0, note="analytical identity at zero cash yield; no third ledger"))


def fold_facts_check(spec, days):
    folds = year_folds(days)
    keys = ("entry_year", "fit_cutoff_index", "fit_cutoff_session", "first_entry_session", "last_entry_session", "first_decision_session", "last_decision_session", "decision_count", "train_decisions", "train_rows", "train_first_decision_index", "train_last_decision_index", "train_max_label_available_session")
    if len(folds) != len(spec["folds"]):
        raise InvalidInput("fold count differs from spec")
    for a, b in zip(folds, spec["folds"]):
        bad = {k: (a[k], b[k]) for k in keys if a[k] != b[k]}
        if bad:
            raise InvalidInput(f"fold {a['entry_year']} differs from spec: {bad}")
    c = spec["counts"]
    if sum(f["train_rows"] for f in folds) != c["total_training_rows_across_fits"] or sum(f["decision_count"] for f in folds) != c["decision_predictions"] or len(folds) != c["real_model_fits"]:
        raise InvalidInput("fold totals differ from spec counts")
    start, end = folds[0]["fit_cutoff_index"], len(days) - 1
    if days[start] != spec["period"][0] or days[end] != spec["period"][1] or end - start + 1 != c["daily_rows_per_ledger"] or (end - start) != c["execution_opportunities_per_ledger"]:
        raise InvalidInput("ledger period differs from spec")
    return folds, start, end


def run(spec_path=SPEC, out_path=None):
    spec = load_spec(spec_path)
    check_parents(spec)
    out = out_path or os.path.join(OUT_DIR, "timing_actionvalue_v1_results.json")
    if os.path.exists(out):
        raise InvalidInput(f"{out} exists; refusing before any fit or ledger")
    days, O, Cr, Ca = load_prices(spec)
    folds, start, end = fold_facts_check(spec, days)
    FIT_CALLS[0] = 0
    res = compute(days, O, Cr, Ca, start, end, folds)
    c = spec["counts"]
    for k in ("M1", "R1"):
        m = res[k]["metrics"]
        if m["unique_daily_rows"] != c["daily_rows_per_ledger"] or m["execution_opportunities"] != c["execution_opportunities_per_ledger"] or m["deposit_events"] != c["deposit_events_per_ledger"] or m["deposits"] != c["deposits_won_per_ledger"]:
            raise InvalidInput(f"ledger counts differ from spec for {k}")
    if len(res["M1"]["decisions"]) != c["decision_predictions"] or res["fit_calls"] != c["real_model_fits"] or sum(m["n_rows"] for m in res["models"].values()) != c["total_training_rows_across_fits"] or res["ledger_calls"] != c["ledger_paths"]:
        raise InvalidInput("decision/fit/row counts differ from spec")
    check_parents(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, res, code_sha, datetime.now(timezone.utc))
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "xb") as f:
        f.write(data)
    print("\n".join(payload["lines"]))
    print("saved", out, "sha", sha_full(out)[:16], "artifact", payload["envelope"]["artifact_id"], "code", code_sha[:16], "spec", spec["sha256"][:16])
    return payload


def synthetic_days(n, start=date(2003, 1, 1)):
    return [(start + timedelta(days=k)).isoformat() for k in range(n)]


def expect_reject(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except InvalidInput:
        return
    raise AssertionError("expected rejection")


def exact_ridge(X, y, alpha):
    Xf = [[Fraction(v) for v in row] for row in X]
    yf = [Fraction(v) for v in y]
    n, p = len(Xf), len(Xf[0])
    mu = [sum(r[k] for r in Xf) / n for k in range(p)]
    var = [sum((r[k] - mu[k]) ** 2 for r in Xf) / n for k in range(p)]
    sd = [math.sqrt(float(v)) if v > 0 else 1.0 for v in var]
    Z = [[Fraction(1)] + [(r[k] - mu[k]) / Fraction(sd[k]) for k in range(p)] for r in Xf]
    q = p + 1
    A = [[sum(Z[i][a] * Z[i][b] for i in range(n)) + (Fraction(alpha) if (a == b and a > 0) else 0) for b in range(q)] for a in range(q)]
    rhs = [sum(Z[i][a] * yf[i] for i in range(n)) for a in range(q)]
    M = [row[:] + [rhs[a]] for a, row in enumerate(A)]
    for col in range(q):
        piv = next(r for r in range(col, q) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(q):
            if r != col and M[r][col] != 0:
                fac = M[r][col]
                M[r] = [a_ - fac * b_ for a_, b_ in zip(M[r], M[col])]
    return [float(M[a][q]) for a in range(q)]


def selftest():
    assert fee(10000) == 1 and fee(20000) == 3 and fee(6666) == 0 and fee(6667) == 1
    O = [0, 5000] + [0] * 19 + [5000]
    assert targets(O, 0) == (-10 / 5000, 0.0)
    O[-1] = 5005
    assert targets(O, 0) == (-5 / 5000, 5 / 5000)
    O = [0, 20000] + [0] * 19 + [21000]
    bP, sP, sQ = buy_cost(20000), sell_proceeds(20000), sell_proceeds(21000)
    assert bP == 20008 and sP == 19993 and sQ == 20992 and targets(O, 0) == ((sQ - bP) / 20000, (sQ - sP) / 20000)
    O[-1] = 19000
    t0, t1 = targets(O, 0)
    assert t0 < 0 and t1 < 0 and abs((t0 - t1) - (sP - bP) / 20000) < 1e-18
    one = fit_ridge([[-1.0], [1.0]], [-2.0, 2.0])
    assert abs(one["beta"][0]) < 1e-12 and abs(one["beta"][1] - 4 / 3) < 1e-12 and one["sd"] == [1.0]
    const = fit_ridge([[-1.0, 7.0], [1.0, 7.0], [-1.0, 7.0], [1.0, 7.0]], [-2.0, 2.0, -2.0, 2.0])
    assert const["sd"][1] == 1.0 and abs(const["beta"][2]) < 1e-12 and abs(const["beta"][1] - 8 / 5) < 1e-12
    expect_reject(fit_ridge, [[1.0]], [1.0])
    n = 420
    days = synthetic_days(n)
    rng = np.random.default_rng(3)
    Cadj = list(1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))))
    Oraw = [int(round(c)) * 10 for c in Cadj]
    Craw = [o + 5 * int(rng.integers(-3, 4)) for o in Oraw]
    Oi, Ci, Ca = validate_series(days, Oraw, Craw, Cadj)
    f = features(Ca, 250)
    assert abs(f[0] - math.log(Ca[250] / Ca[245])) < 1e-15 and abs(f[1] - math.log(Ca[250] / Ca[230])) < 1e-15 and abs(f[2] - math.log(Ca[250] / Ca[190])) < 1e-15 and abs(f[3] - (Ca[250] / (sum(Ca[51:251]) / 200) - 1)) < 1e-12 and features(Ca, FEATURE_START - 1) is None
    r20 = [math.log(Ca[t] / Ca[t - 1]) for t in range(231, 251)]
    assert abs(f[4] - float(np.std(r20))) < 1e-15 and len(r20) == 20
    Ca_f = list(Ca)
    for t in range(251, n):
        Ca_f[t] *= 1.5
    assert features(Ca_f, 250) == f
    cutoff = 330
    X, y, used = training_rows(Oi, Ca, cutoff)
    assert used[0] == FEATURE_START and used[-1] == cutoff - H - 1 and (cutoff - H) not in used and all(j + H + 1 <= cutoff for j in used) and len(X) == 2 * len(used) and X[0][5] == 0.0 and X[1][5] == 1.0 and X[0][:5] == X[1][:5]
    assert y[0] == targets(Oi, used[0])[0] and y[1] == targets(Oi, used[0])[1]
    mdl = fit_ridge(X, y)
    ex = exact_ridge(X[:40], y[:40], ALPHA)
    m40 = fit_ridge(X[:40], y[:40])
    assert all(abs(a - b) <= 1e-9 + 1e-7 * abs(b) for a, b in zip(m40["beta"], ex)), (m40["beta"], ex)
    Zchk = np.column_stack([np.ones(len(y)), (np.asarray(X) - np.asarray(mdl["mu"])) / np.asarray(mdl["sd"])])
    pen = np.eye(7) * ALPHA
    pen[0, 0] = 0
    assert np.allclose(Zchk.T @ (np.asarray(y) - Zchk @ np.asarray(mdl["beta"])), pen @ np.asarray(mdl["beta"]), atol=1e-9)
    O_p, Ca_p = list(Oi), list(Ca)
    for t in range(cutoff + 1, n):
        O_p[t] += 500
        Ca_p[t] *= 1.3
    X2, y2, used2 = training_rows(O_p, Ca_p, cutoff)
    assert X2 == X and y2 == y and used2 == used and fit_ridge(X2, y2)["beta"] == mdl["beta"]
    assert training_rows(Oi, Ca, FEATURE_START + H)[2] == [] and training_rows(Oi, Ca, FEATURE_START + H + 1)[2] == [FEATURE_START]
    d5 = synthetic_days(5)
    O5, C5 = [5000] * 5, [5000] * 5
    seq = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0}
    led = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="x", w=seq[i]))
    assert [(x["side"], x["qty"], x["price"], x["fee"]) for x in led["trades"]] == [("buy", 1, 5005, 0), ("buy", 2, 5005, 1), ("sell", 1, 4995, 0), ("buy", 5, 5005, 3)]
    assert led["metrics"]["final_cash"] == 14951 and led["metrics"]["qty_end"] == 7 and led["metrics"]["pnl_won_after_costs"] == -49 and led["metrics"]["fees_total_won"] == 4 and led["metrics"]["tick_cost_won"] == 45 and led["metrics"]["execution_opportunities"] == 4 and led["metrics"]["flat_to_long_entries"] == 1 and led["metrics"]["long_maintenance_buys"] == 2
    one_s = ledger_run(d5[:1], O5[:1], C5[:1], 0, 0, lambda i, held: dict(action="long", w=1.0))
    assert one_s["executions"] == [] and one_s["decisions"] == [] and one_s["metrics"]["final_value"] == DEPOSIT
    two_s = ledger_run(d5[:2], O5[:2], C5[:2], 0, 1, lambda i, held: dict(action="long", w=1.0))
    assert len(two_s["executions"]) == 1 and two_s["executions"][0]["realized_qty"] == DEPOSIT // 5005 and two_s["daily"][1]["w_for_next_session"] is None
    seqf = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}
    ff = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="x", w=seqf[i]))
    assert [t["kind"] for t in ff["trades"]] == ["entry", "exit", "entry"] and ff["metrics"]["long_to_flat_exits"] == 1 and ff["metrics"]["flat_to_long_entries"] == 2 and ff["daily"][2]["held"] == 0 and ff["executions"][2]["side"] is None
    keep = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="keep", w=float(held)))
    assert keep["metrics"]["trades"] == 0 and all(d["held"] == 0 for d in keep["daily"])
    poor = ledger_run(d5, [50000] * 5, [50000] * 5, 0, 4, lambda i, held: dict(action="long", w=1.0))
    assert poor["metrics"]["trades"] == 0 and all(d["held"] == 0 for d in poor["daily"]) and all(e["side"] is None for e in poor["executions"])
    d6 = synthetic_days(6)
    poor6 = ledger_run(d6, [45000] * 6, [45000] * 6, 0, 5, lambda i, held: dict(action="long", w=1.0))
    assert poor6["metrics"]["trades"] == 1 and poor6["trades"][0]["execution_session"] == d6[5] and poor6["trades"][0]["qty"] == 1 and poor6["daily"][4]["held"] == 0 and poor6["daily"][5]["held"] == 1
    big_days = synthetic_days(1300, start=date(2011, 10, 1))
    rng2 = np.random.default_rng(5)
    Cb = list(1000.0 * np.exp(np.cumsum(rng2.normal(0.0003, 0.012, 1300))))
    Ob = [int(round(c)) * 10 for c in Cb]
    Cbr = [o + 5 * int(rng2.integers(-3, 4)) for o in Ob]
    Obi, Cbi, Cba = validate_series(big_days, Ob, Cbr, Cb)
    folds = year_folds(big_days, 2013, 2015)
    for fd in folds:
        c = fd["fit_cutoff_index"]
        assert big_days[c + 1] >= f"{fd['entry_year']}-01-01" > big_days[c] and fd["train_last_decision_index"] == c - H - 1 and fd["train_rows"] == 2 * fd["train_decisions"]
    assert folds[-1]["last_decision_index"] == 1298 and folds[0]["first_decision_index"] == folds[0]["fit_cutoff_index"] and folds[1]["first_decision_index"] == folds[0]["last_decision_index"] + 1
    start_b = folds[0]["fit_cutoff_index"]
    FIT_CALLS[0] = 0
    res = compute(big_days, Obi, Cbi, Cba, start_b, 1299, folds)
    assert res["ledger_calls"] == 2 and res["fit_calls"] == 3 and len(res["models"]) == 3 and res["M1"]["metrics"]["sessions"] == 1300 - start_b and len(res["M1"]["decisions"]) == 1299 - start_b
    for d in res["M1"]["decisions"]:
        y = d["fit_entry_year"]
        assert d["entry_session"][:4] == str(y) and d["features"]["held"] == d["held_at_close"]
    for y, mm in res["models"].items():
        fd = next(f for f in folds if f["entry_year"] == y)
        assert mm["fit_cutoff_index"] == fd["fit_cutoff_index"] and mm["train_max_label_available_session"] == fd["train_max_label_available_session"] and mm["train_max_label_available_session"] <= mm["fit_cutoff_session"]
    assert sum(res["yearly"]["M1"].values()) == res["M1"]["metrics"]["pnl_won_after_costs"] and sum(res["yearly"]["M1_minus_R1"].values()) == res["paired"]["m1_minus_r1_terminal_pnl"]
    assert all(d["w"] == 1.0 for d in res["R1"]["decisions"]) and res["R1"]["metrics"]["sells"] == 0 and res["R1"]["metrics"]["execution_opportunities"] == len(res["M1"]["decisions"])
    assert all(isinstance(d["cash"], int) and d["cash"] >= 0 and isinstance(d["qty"], int) for d in res["M1"]["daily"])
    Ob_f = list(Obi)
    for t in range(folds[1]["fit_cutoff_index"] + 1, 1300):
        Ob_f[t] += 500
    res_f = compute(big_days, Ob_f, Cbi, Cba, start_b, 1299, folds)
    assert res_f["models"][2013]["beta"] == res["models"][2013]["beta"] and res_f["models"][2014]["beta"] == res["models"][2014]["beta"] and res_f["models"][2015]["beta"] != res["models"][2015]["beta"]
    fixed = {}
    for i in range(start_b, 1299):
        fixed[i] = 1.0 if (i // 7) % 3 else -1.0
    fx = ledger_run(big_days, Obi, Cbi, start_b, 1299, lambda i, held: dict(action="long" if fixed[i] > 0 else "flat", w=1.0 if fixed[i] > 0 else 0.0))
    assert fx["metrics"]["long_to_flat_exits"] > 0 and fx["metrics"]["flat_to_long_entries"] > 0 and fx["metrics"]["long_maintenance_buys"] > 0 and fx["daily"][-1]["w_for_next_session"] is None and fx["executions"][-1]["execution_session"] == big_days[1299]
    lines = summary_lines(res)
    assert "M1−R1" in lines[3]
    expect_reject(validate_series, ["a", "b"], [100, 100], [100], [1.0, 1.0])
    expect_reject(validate_series, ["b", "a"], [100, 100], [100, 100], [1.0, 1.0])
    expect_reject(validate_series, ["a", "b"], [100.5, 100], [100, 100], [1.0, 1.0])
    expect_reject(validate_series, ["a", "b"], [100, 100], [100, 100], [1.0, float("nan")])
    expect_reject(predict, dict(mu=[0] * 6, sd=[1] * 6, beta=[float("inf")] + [0] * 6), [0] * 5, 0)
    d0, d1 = date(2002, 10, 14), date.fromisoformat(CUTOFF)
    span = (d1 - d0).days
    sd_full = [(d0 + timedelta(days=round(k * span / (RETAINED_ROWS - 1)))).isoformat() for k in range(RETAINED_ROWS)] + ["2026-11-06"]
    assert len(sd_full) == FULL_ROWS and sd_full == sorted(sd_full) and len(set(sd_full)) == FULL_ROWS and sd_full[0] == "2002-10-14" and sd_full[RETAINED_ROWS - 1] == CUTOFF
    ok_raw = dict(days=list(sd_full), O=[20000] * FULL_ROWS, C=[20000] * FULL_ROWS)
    ok_adj = dict(days=list(sd_full), C=[20000.0] * FULL_ROWS)
    pd_, pO, pCr, pCa = prepare_prices(ok_raw, ok_adj)
    assert len(pd_) == RETAINED_ROWS and pd_[-1] == CUTOFF
    bad_adj = dict(ok_adj)
    bad_adj["days"] = list(sd_full[:-1]) + ["2026-11-09"]
    expect_reject(prepare_prices, ok_raw, bad_adj)
    bad_raw = dict(ok_raw)
    bad_raw["days"] = list(sd_full[:-1]) + [sd_full[-2]]
    expect_reject(prepare_prices, bad_raw, ok_adj)
    expect_reject(prepare_prices, dict(ok_raw, O=ok_raw["O"][:-1]), ok_adj)
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 5
    import tempfile
    tmp = tempfile.NamedTemporaryFile(prefix="timing_existing_", suffix=".json", delete=False)
    tmp.write(b"{}")
    tmp.close()
    f0, l0 = FIT_CALLS[0], LEDGER_CALLS[0]
    try:
        expect_reject(run, SPEC, tmp.name)
        assert FIT_CALLS[0] == f0 and LEDGER_CALLS[0] == l0
    finally:
        os.unlink(tmp.name)
    rdays, rO, rCr, rCa = load_prices(spec)
    rfolds, rstart, rend = fold_facts_check(spec, rdays)
    assert len(rfolds) == 12 and rstart == 3031 and rend == 5937 and sum(f["train_rows"] for f in rfolds) == 99940 and sum(f["decision_count"] for f in rfolds) == 2906
    assert check_parents(spec) == 5
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], real_data="hashes, shapes/calendar, fold metadata and counts only; no real features, labels, fits, predictions or ledgers"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: timing_v3.py | selftest")
