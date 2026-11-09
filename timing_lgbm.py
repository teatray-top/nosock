import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import importlib.metadata as md
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import lightgbm as lgb

SPEC = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\timing_lgbm_v1_spec_20261106.json")
SPEC_SHA256 = "b0c7e7b43fa54aae24d3d5b29304a9623aa1c3a8ade18263a6c84da8ce90d48e"
OUT_DIR = os.path.expanduser(r"~\Documents\Codex\2026-11-05\x20-teatray-top-irc\work\timing_lgbm_v1_20261106")
SCHEMA = "timing-lgbm-v1"
MODULE = "timing_lgbm_v1"
SYM = "069500"
CUTOFF = "2026-11-05"
FULL_ROWS, RETAINED_ROWS = 5939, 5938
H = 20
FEATURE_START = 199
FIRST_EXEC_YEAR, LAST_EXEC_YEAR = 2015, 2026
DEPOSIT = 10000
TICK = 5
FEE_NUM, FEE_DEN = 15, 100000
FEATURES = ("logret5", "logret20", "logret60", "ma200_distance", "sd20", "held")
PARENT_ORDER = ("raw_prices", "adjusted_prices", "allocation_mechanics_spec", "module_contract", "prior_timing_spec", "frozen_M1_R1_result", "timing_lgbm_spec")
NUM_ROUNDS = 200
PARAMS = dict(objective="regression", boosting="gbdt", device_type="cpu", num_leaves=7, max_depth=3, learning_rate=0.05, min_data_in_leaf=200, lambda_l1=0.0, lambda_l2=1.0, min_sum_hessian_in_leaf=0.001, min_gain_to_split=0.0, feature_fraction=1.0, feature_fraction_bynode=1.0, bagging_fraction=1.0, bagging_freq=0, seed=0, data_random_seed=0, feature_fraction_seed=0, bagging_seed=0, num_threads=1, deterministic=True, force_col_wise=True, force_row_wise=False, max_bin=255, min_data_in_bin=3, bin_construct_sample_cnt=200000, boost_from_average=True, feature_pre_filter=True, zero_as_missing=False, use_missing=True, linear_tree=False, metric="None", verbosity=-1)
LGBM_VERSION = "4.6.0"
LEDGER_CALLS = [0]
FIT_CALLS = [0]


class InvalidInput(SystemExit):
    pass


def sha_full(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def environment_facts():
    dll = os.path.join(os.path.dirname(lgb.__file__), "bin", "lib_lightgbm.dll")
    facts = dict(python_executable=sys.executable, python_version=sys.version.split()[0], lightgbm=lgb.__version__, numpy=np.__version__, native_binary_path=dll, native_binary_sha256=sha_full(dll) if os.path.exists(dll) else None, thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    for pkg in ("scipy", "scikit-learn"):
        try:
            facts[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            facts[pkg] = None
    return facts


def load_spec(path=SPEC):
    h = sha_full(path)
    if h != SPEC_SHA256:
        raise InvalidInput(f"spec hash {h} != {SPEC_SHA256}; refusing")
    s = json.load(open(path, encoding="utf-8"))
    m = s["model"]
    if s["version"] != SCHEMA or s["output"]["directory"] != OUT_DIR or s["resources"]["cpu_threads"] != 1 or list(s["parent_order"]) != list(PARENT_ORDER):
        raise InvalidInput("spec settings differ from module constants")
    if m["num_boost_round"] != NUM_ROUNDS or m["params"] != PARAMS or m["versions"]["lightgbm"] != LGBM_VERSION or s["action_values"]["N"] != H or list(s["features"]["order"]) != list(FEATURES) or s["data"]["full_input_rows"] != FULL_ROWS or s["data"]["retained_rows"] != RETAINED_ROWS:
        raise InvalidInput("model/data constants differ from module constants")
    parents = {k: dict(path=s["parents"][k]["path"], sha256=s["parents"][k]["sha256"]) for k in ("raw_prices", "adjusted_prices", "allocation_mechanics_spec", "module_contract", "prior_timing_spec", "frozen_M1_R1_result")}
    parents["timing_lgbm_spec"] = dict(path=path, sha256=SPEC_SHA256)
    return dict(path=path, sha256=h, parents=parents, folds=s["walk_forward"]["folds"], counts=s["counts"], period=s["ledger"]["period"], native_binary_sha256=m["native_binary"]["sha256"], versions=m["versions"], external_timeout_seconds=int(s["resources"]["external_timeout_seconds"]))


def check_parents(spec):
    for name in PARENT_ORDER:
        p = spec["parents"][name]
        if not os.path.exists(p["path"]):
            raise InvalidInput(f"parent {name} missing")
        h = sha_full(p["path"])
        if h != p["sha256"]:
            raise InvalidInput(f"parent {name} hash changed: {h}")
    return len(PARENT_ORDER)


def check_environment(spec):
    f = environment_facts()
    if f["lightgbm"] != spec["versions"]["lightgbm"] or f["numpy"] != spec["versions"]["numpy"]:
        raise InvalidInput(f"library versions differ from spec: {f['lightgbm']}, {f['numpy']}")
    if f["native_binary_sha256"] != spec["native_binary_sha256"]:
        raise InvalidInput("native LightGBM binary hash differs from spec")
    return f


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
    keep = [k for k, d in enumerate(raw["days"]) if d <= CUTOFF]
    rd = [raw["days"][k] for k in keep]
    if len(rd) != RETAINED_ROWS or rd[-1] != CUTOFF or rd[0] != "2002-10-14":
        raise InvalidInput("retained calendar differs from the fixed facts")
    O, Cr, Ca = validate_series(rd, [raw["O"][k] for k in keep], [raw["C"][k] for k in keep], [adj["C"][k] for k in keep])
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
    firsts = {y: next((k for k, d in enumerate(days) if d >= f"{y}-01-01"), None) for y in range(first_year, last_year + 2)}
    out = []
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


def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def fit_lgbm(X, y):
    FIT_CALLS[0] += 1
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(FEATURES) or X.shape[0] != y.shape[0] or X.shape[0] < 2:
        raise InvalidInput("training matrix shape invalid")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise InvalidInput("non-finite training data")
    ds = lgb.Dataset(X, label=y, feature_name=list(FEATURES), categorical_feature=[], reference=None, free_raw_data=False, params=dict(PARAMS))
    booster = lgb.train(dict(PARAMS), ds, num_boost_round=NUM_ROUNDS)
    n_trees = int(booster.num_trees())
    cur_iter = int(booster.current_iteration())
    model_str = booster.model_to_string(num_iteration=-1)
    dump = booster.dump_model(num_iteration=-1)
    dump_bytes = canonical_json_bytes(dump)
    pred_train = booster.predict(X, start_iteration=0, num_iteration=-1, num_threads=1)
    reloaded = lgb.Booster(model_str=model_str)
    pred_reload = reloaded.predict(X, start_iteration=0, num_iteration=-1, num_threads=1)
    parity = float(np.max(np.abs(pred_train - pred_reload))) if len(X) else 0.0
    if not np.isfinite(pred_train).all() or parity > 1e-9:
        raise InvalidInput(f"model predictions non-finite or reload parity failed ({parity})")
    resid = y - pred_train
    var_y = float(np.var(y))
    if var_y <= (1e-12 * abs(float(np.mean(y)))) ** 2:
        var_y = 0.0
    gain = booster.feature_importance(importance_type="gain")
    return dict(booster=booster, model_string=model_str, model_string_sha256=sha_bytes(model_str.encode("utf-8")), dump=dump, dump_sha256=sha_bytes(dump_bytes), dump_encoding="canonical JSON: sort_keys, separators (',',':'), ensure_ascii False, UTF-8", num_trees=n_trees, current_iteration=cur_iter, requested_rounds=NUM_ROUNDS, params=dict(PARAMS), feature_names=list(FEATURES), n_rows=int(len(y)), target_mean=float(np.mean(y)), target_var=var_y, residual_variance_ratio=(float(np.var(resid) / var_y) if var_y > 0 else None), residual_variance_ratio_note="population variance of training residuals / population variance of training targets" if var_y > 0 else "null: zero target variance", gain_importance={n: float(g) for n, g in zip(FEATURES, gain)}, reload_parity_max_abs_error=parity)


def predict_model(model, feats, held):
    x = np.asarray([list(feats) + [float(held)]], dtype=np.float64)
    val = float(model["booster"].predict(x, start_iteration=0, num_iteration=-1, num_threads=1)[0])
    if not math.isfinite(val):
        raise InvalidInput("non-finite prediction")
    return val


def traverse_dump(dump, x):
    total = 0.0
    for tree in dump["tree_info"]:
        node = tree["tree_structure"]
        while "leaf_value" not in node:
            v = x[node["split_feature"]]
            thr = node["threshold"]
            if node.get("decision_type", "<=") != "<=":
                raise InvalidInput("unexpected decision type in dump")
            node = node["left_child"] if v <= thr else node["right_child"]
        total += node["leaf_value"]
    return total


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


def yearly_increments(pnl_by_session):
    last = {}
    for s, v in pnl_by_session:
        last[s[:4]] = v
    out, prev = {}, 0
    for y in sorted(last):
        out[y] = last[y] - prev
        prev = last[y]
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
        return dict(count=0, min=None, median=None, max=None, mean=None)
    s = sorted(runs)
    k = len(s)
    med = float(s[k // 2]) if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2.0
    return dict(count=k, min=s[0], median=med, max=s[-1], mean=sum(s) / k)


def exit_episodes(trades):
    eps, open_exit = [], None
    for t in trades:
        if t["kind"] == "exit":
            open_exit = t["execution_session"]
        elif t["kind"] == "entry" and open_exit is not None:
            eps.append(dict(exit_session=open_exit, reentry_session=t["execution_session"]))
            open_exit = None
    if open_exit is not None:
        eps.append(dict(exit_session=open_exit, reentry_session=None))
    return eps


def build_models(days, O, C_adj, folds):
    models = {}
    for fd in folds:
        c = fd["fit_cutoff_index"]
        X, yv, used = training_rows(O, C_adj, c)
        if len(used) != fd["train_decisions"] or len(yv) != fd["train_rows"] or used[0] != fd["train_first_decision_index"] or used[-1] != fd["train_last_decision_index"]:
            raise InvalidInput(f"training rows differ from fold facts for {fd['entry_year']}")
        m = fit_lgbm(X, yv)
        m.update(entry_year=fd["entry_year"], fit_cutoff_index=c, fit_cutoff_session=days[c], first_entry_session=fd["first_entry_session"], train_decisions=len(used), train_first_decision_session=days[used[0]], train_last_decision_session=days[used[-1]], train_max_label_available_session=days[used[-1] + H + 1])
        models[fd["entry_year"]] = m
    return models


def make_decider(days, C_adj, models, folds):
    by_dec = {}
    for fd in folds:
        for i in range(fd["first_decision_index"], fd["last_decision_index"] + 1):
            by_dec[i] = fd["entry_year"]

    def decide(i, held):
        y = by_dec.get(i)
        if y is None or y not in models:
            raise InvalidInput(f"no model for decision {days[i]}")
        f = features(C_adj, i)
        pred = predict_model(models[y], f, held)
        if pred > 0:
            w, action = 1.0, "long"
        elif pred < 0:
            w, action = 0.0, "flat"
        else:
            w, action = float(held), "keep"
        return dict(fit_entry_year=y, fit_cutoff_session=models[y]["fit_cutoff_session"], train_max_label_available_session=models[y]["train_max_label_available_session"], features=dict(zip(FEATURES, f + [held])), predicted_contrast=pred, action=action, w=w)

    return decide


def anchor_from_obj(obj, expected_sha=None, spec=None):
    if expected_sha is not None and obj.get("_sha256") != expected_sha:
        raise InvalidInput("anchor hash mismatch")
    env = obj["envelope"]
    if env.get("schema_version") != "timing-actionvalue-v1" or env.get("symbol") != SYM:
        raise InvalidInput("anchor schema/symbol mismatch")
    if spec is not None and env.get("spec_sha256") != spec["parents"]["prior_timing_spec"]["sha256"]:
        raise InvalidInput("anchor spec hash differs from the prior timing spec parent")
    out = {}
    for k in ("M1", "R1"):
        v = obj["variants"][k]
        met = v["metrics"]
        daily = v["daily_records"]
        sessions = [d["session"] for d in daily]
        if len(daily) != len(set(sessions)) or sessions != sorted(sessions):
            raise InvalidInput(f"anchor {k} daily rows not unique/sorted")
        if met["deposits"] != len(daily) * DEPOSIT or met["unique_daily_rows"] != len(daily):
            raise InvalidInput(f"anchor {k} deposits/rows inconsistent")
        out[k] = dict(metrics=met, pnl_by_session=[(d["session"], d["pnl"]) for d in daily], qty_by_session={d["session"]: d["qty"] for d in daily}, first=sessions[0], last=sessions[-1], rows=len(daily))
    return dict(artifact_id=env["artifact_id"], spec_sha256=env.get("spec_sha256"), variants=out)


def load_anchor(spec):
    p = spec["parents"]["frozen_M1_R1_result"]
    h = sha_full(p["path"])
    if h != p["sha256"]:
        raise InvalidInput("anchor hash changed")
    obj = json.load(open(p["path"], encoding="utf-8"))
    obj["_sha256"] = h
    return anchor_from_obj(obj, expected_sha=h, spec=spec)


def anchor_calendar_check(anchor, days, start, end, counts):
    for k, v in anchor["variants"].items():
        if v["first"] != days[start] or v["last"] != days[end] or v["rows"] != counts["daily_rows_per_ledger"] or v["metrics"]["deposits"] != counts["deposits_won_per_ledger"] or v["metrics"]["execution_opportunities"] != counts["execution_opportunities_per_ledger"]:
            raise InvalidInput(f"anchor {k} period/counts differ from spec")
        if [s for s, _ in v["pnl_by_session"]] != days[start:end + 1]:
            raise InvalidInput(f"anchor {k} daily sessions differ from the calendar")
    return True


def compare(M2, anchor):
    out = {}
    m2 = {d["session"]: d for d in M2["daily"]}
    for k, v in anchor["variants"].items():
        diffs = [m2[s]["pnl"] - p for s, p in v["pnl_by_session"]]
        inc_a = yearly_increments(v["pnl_by_session"])
        inc_m2 = yearly_increments([(d["session"], d["pnl"]) for d in M2["daily"]])
        out[f"M2_minus_{k}"] = dict(role="primary" if k == "R1" else "secondary", terminal_pnl=M2["metrics"]["pnl_won_after_costs"] - v["metrics"]["pnl_won_after_costs"], max_dd_won_diff=M2["metrics"]["max_dd_won"] - v["metrics"]["max_dd_won"], daily_diff_min=min(diffs), daily_diff_max=max(diffs), sessions_m2_fewer_shares=sum(1 for s, q in v["qty_by_session"].items() if m2[s]["qty"] < q), yearly_increments_diff={y: inc_m2[y] - inc_a[y] for y in inc_m2}, fees_diff=M2["metrics"]["fees_total_won"] - v["metrics"]["fees_total_won"], tick_diff=M2["metrics"]["tick_cost_won"] - v["metrics"]["tick_cost_won"], turnover_diff=M2["metrics"]["turnover_won"] - v["metrics"]["turnover_won"], exposure_diff=M2["metrics"]["exposure_avg"] - v["metrics"]["exposure_avg"], counts=dict(M2=dict(buys=M2["metrics"]["buys"], sells=M2["metrics"]["sells"], entries=M2["metrics"]["flat_to_long_entries"], exits=M2["metrics"]["long_to_flat_exits"], maintenance=M2["metrics"]["long_maintenance_buys"]), anchor=dict(buys=v["metrics"]["buys"], sells=v["metrics"]["sells"], entries=v["metrics"]["flat_to_long_entries"], exits=v["metrics"]["long_to_flat_exits"], maintenance=v["metrics"]["long_maintenance_buys"])))
    return out


def compute(days, O, C_raw, C_adj, start, end, folds):
    models = build_models(days, O, C_adj, folds)
    LEDGER_CALLS[0] = 0
    M2 = ledger_run(days, O, C_raw, start, end, make_decider(days, C_adj, models, folds))
    sign = {}
    for d in M2["decisions"]:
        y = d["entry_session"][:4]
        g = sign.setdefault(y, dict(n=0, pred_pos=0, pred_neg=0, pred_zero=0, desired_long=0))
        g["n"] += 1
        g["pred_pos"] += int(d["predicted_contrast"] > 0)
        g["pred_neg"] += int(d["predicted_contrast"] < 0)
        g["pred_zero"] += int(d["predicted_contrast"] == 0)
        g["desired_long"] += int(d["w"] == 1.0)
    return dict(models=models, M2=M2, sign=sign, yearly_M2=yearly_increments([(d["session"], d["pnl"]) for d in M2["daily"]]), streaks=holding_streaks(M2["daily"]), episodes=exit_episodes(M2["trades"]), ledger_calls=LEDGER_CALLS[0], fit_calls=FIT_CALLS[0])


def summary_lines(res, anchor, cmp):
    m = res["M2"]["metrics"]
    lines = [f"[{m['start']}~{m['end']} {m['sessions']}세션 입금 {m['deposits']:,} 적합 {len(res['models'])}회 원장 {res['ledger_calls']}개 결정 {len(res['M2']['decisions'])} 앵커 로드 {len(anchor['variants'])}]"]
    lines.append(f"  M2 LightGBM: 손익 {m['pnl_won_after_costs']:+,} ({m['pnl_pct_of_deposits']*100:+.1f}%) 낙폭 {m['max_dd_won']:+,} 노출 {m['exposure_avg']*100:.0f}% 보유일 {m['days_held']} 진입 {m['flat_to_long_entries']} 청산 {m['long_to_flat_exits']} 유지매수 {m['long_maintenance_buys']} 매수 {m['buys']} 매도 {m['sells']} 수수료 {m['fees_total_won']:,} 틱 {m['tick_cost_won']:,} 회전 {m['turnover_won']:,} 기말 현금 {m['final_cash']:,} 수량 {m['qty_end']} 가치 {m['final_value']:,}")
    for k, v in anchor["variants"].items():
        am = v["metrics"]
        lines.append(f"  앵커 {k}(로드): 손익 {am['pnl_won_after_costs']:+,} 낙폭 {am['max_dd_won']:+,} 보유일 {am['days_held']} 진입 {am['flat_to_long_entries']} 청산 {am['long_to_flat_exits']} 수수료 {am['fees_total_won']:,} 틱 {am['tick_cost_won']:,}")
    for k, c in cmp.items():
        lines.append(f"  {k} [{c['role']}]: 기말 {c['terminal_pnl']:+,} 낙폭차 {c['max_dd_won_diff']:+,} 일별 최소 {c['daily_diff_min']:+,} 최대 {c['daily_diff_max']:+,} 보유주 적은 세션 {c['sessions_m2_fewer_shares']} 수수료차 {c['fees_diff']:+,} 틱차 {c['tick_diff']:+,} | 연도별 증분차: " + ", ".join(f"{y} {x:+,}" for y, x in c["yearly_increments_diff"].items()))
    lines.append("  연도별 증분 M2: " + ", ".join(f"{y} {v:+,}" for y, v in res["yearly_M2"].items()))
    lines.append("  예측 부호(진입연도): " + ", ".join(f"{y} +{g['pred_pos']}/−{g['pred_neg']}/0:{g['pred_zero']}" for y, g in sorted(res["sign"].items())))
    lines.append(f"  보유 연속구간: {res['streaks']} | 청산→재진입 구간: {res['episodes']}")
    for y, md_ in res["models"].items():
        rvr = md_["residual_variance_ratio"]
        lines.append(f"  fit {y}: cutoff {md_['fit_cutoff_session']} 행 {md_['n_rows']} 트리 {md_['num_trees']}/{md_['requested_rounds']} 잔차분산비 {rvr if rvr is None else f'{rvr:.4f}'} 재로드오차 {md_['reload_parity_max_abs_error']:.1e} gain " + ", ".join(f"{k}:{v:.3e}" for k, v in md_["gain_importance"].items()))
    return lines


def build_payload(spec, envf, res, anchor, cmp, code_sha, now_utc):
    artifact_id = f"{MODULE}_{now_utc.strftime('%Y%m%d_%H%M%S')}_{code_sha[:8]}"
    p = spec["parents"]
    env = dict(schema_version=SCHEMA, artifact_id=artifact_id, parent_artifact_hashes=[p[k]["sha256"] for k in PARENT_ORDER], parent_artifact_roles={k: dict(path=p[k]["path"], sha256=p[k]["sha256"]) for k in PARENT_ORDER}, anchor_artifact_id=anchor["artifact_id"], code_sha256=code_sha, spec_sha256=spec["sha256"], asof_utc=now_utc.isoformat(timespec="seconds"), environment=envf, exchange_session=dict(exchange="KRX", symbol=SYM, calendar="raw daily session dates", timezone="Asia/Seoul", utc_mapping="session date D in KST (UTC+09:00); no intraday clock times used", decision_marker="session_close", execution_marker="next_session_open"), symbol=SYM, availability_status="available", decision_at="session_close (signed action-value contrast from closes through i and the actual held state; logical marker)", execution_not_before="next_session_open (quantity decided after observing the open; idealized fill at O +/- 5, no actual order)", model=dict(name="LightGBM regression GBDT", api="lightgbm.train", num_boost_round=NUM_ROUNDS, params=dict(PARAMS), features=list(FEATURES), standardization="none (held numeric 0/1)", dataset="one new Dataset per fold from that fold's matured training rows only; reference None; categorical_feature []", output="signed raw regression value; not a probability; no clipping/calibration"), action_values=dict(N=H, entry="P = raw O[i+1]", endpoint="Q = raw O[i+21]", fee="floor(n x 15 / 100000)", buy_cost="b(P) = P + 5 + F(P+5)", net_sell="s(P) = P - 5 - F(P-5)", target_cash="z(i,0) = (s(Q) - b(P)) / P", target_held="z(i,1) = (s(Q) - s(P)) / P", caveat="one-share proxy with no intervening trades; repeated daily policy need not hold 20 sessions; not an optimal dynamic value"), walk_forward=dict(fold_by="calendar year of next-open execution", fits=len(res["models"]), continuous_account="one path 2014-12-30 .. 2026-11-05, no yearly reset", caveat="all historical periods viewed; retrospective walk-forward; no sealed window or prospective validation"), ledger=dict(mechanics="allocate-v1 timeline; integer won; deposit 10,000 at each close; idealized fills; zero tax, zero cash yield, no dividends/distributions", baselines="M1 and R1 loaded from the frozen timing-actionvalue-v1 result; no baseline fit or ledger replay"), units=dict(cash="krw (integer)", value="krw (integer)", pnl="krw (integer)", qty="shares (integer)", w="weight in {0,1}", predicted_contrast="dimensionless per-share value contrast", pnl_pct_of_deposits="fraction, not annualized"), counts=dict(real_model_fits=len(res["models"]), fit_calls=res["fit_calls"], total_training_rows_across_fits=sum(m["n_rows"] for m in res["models"].values()), decision_predictions=len(res["M2"]["decisions"]), ledger_paths=res["ledger_calls"], daily_rows_per_ledger=res["M2"]["metrics"]["unique_daily_rows"], execution_opportunities_per_ledger=res["M2"]["metrics"]["execution_opportunities"], deposit_events_per_ledger=res["M2"]["metrics"]["deposit_events"], deposits_won_per_ledger=res["M2"]["metrics"]["deposits"], frozen_anchor_paths_loaded=len(anchor["variants"]), requested_boosting_rounds_per_fit=NUM_ROUNDS, actual_trees_per_fit={str(y): m["num_trees"] for y, m in res["models"].items()}, optimizer_calls=0, hyperparameter_searches=0, api_calls=0, orders=0), interpretation="annual increments include valuation of carried share differences; a better in-sample residual fit or more sell signals do not establish future profitability; min_data_in_leaf 200 counts rows (two states per date, overlapping labels), not independent dates", claims="no guaranteed profit or live advantage; a nonlinear regression may fit noise and fail to improve timing; failure alone does not diagnose overfitting or unpredictability; the policy is not optimal; no new prospective validation in this run", resources=dict(cpu_threads=1, external_timeout_seconds=spec["external_timeout_seconds"], real_experiments=1, gpu=0, api_calls=0, orders=0))
    ref = dict(envelope_artifact_id=artifact_id)
    fits = {str(y): {k: v for k, v in m.items() if k != "booster"} for y, m in res["models"].items()}
    return dict(envelope=env, lines=summary_lines(res, anchor, cmp), fits=fits, decision_records=[dict(d, **ref) for d in res["M2"]["decisions"]], M2=dict(metrics=res["M2"]["metrics"], execution_records=[dict(e, **ref) for e in res["M2"]["executions"]], trade_records=[dict(t, **ref) for t in res["M2"]["trades"]], daily_records=[dict(d, **ref) for d in res["M2"]["daily"]]), anchors_loaded={k: dict(metrics=v["metrics"], loaded_from=anchor["artifact_id"]) for k, v in anchor["variants"].items()}, comparisons=cmp, annual=dict(M2=res["yearly_M2"]), sign_counts=res["sign"], holding_streaks=res["streaks"], exit_reentry_episodes=res["episodes"])


def verify_serialized_fits(fits):
    for y, m in fits.items():
        if "dump" not in m or "tree_info" not in m["dump"]:
            raise InvalidInput(f"fit {y} lacks the full dump")
        if sha_bytes(canonical_json_bytes(m["dump"])) != m["dump_sha256"]:
            raise InvalidInput(f"fit {y} serialized dump hash differs from the recorded dump hash")
        if len(m["dump"]["tree_info"]) != m["num_trees"]:
            raise InvalidInput(f"fit {y} tree_info length differs from num_trees")
        if sha_bytes(m["model_string"].encode("utf-8")) != m["model_string_sha256"]:
            raise InvalidInput(f"fit {y} model string hash differs")
    return len(fits)


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
    envf = check_environment(spec)
    out = out_path or os.path.join(OUT_DIR, "timing_lgbm_v1_results.json")
    if os.path.exists(out):
        raise InvalidInput(f"{out} exists; refusing before loading prices or computing")
    anchor = load_anchor(spec)
    days, O, Cr, Ca = load_prices(spec)
    folds, start, end = fold_facts_check(spec, days)
    anchor_calendar_check(anchor, days, start, end, spec["counts"])
    FIT_CALLS[0] = 0
    res = compute(days, O, Cr, Ca, start, end, folds)
    c = spec["counts"]
    m = res["M2"]["metrics"]
    if m["unique_daily_rows"] != c["daily_rows_per_ledger"] or m["execution_opportunities"] != c["execution_opportunities_per_ledger"] or m["deposit_events"] != c["deposit_events_per_ledger"] or m["deposits"] != c["deposits_won_per_ledger"] or len(res["M2"]["decisions"]) != c["decision_predictions"] or res["fit_calls"] != c["real_model_fits"] or sum(mm["n_rows"] for mm in res["models"].values()) != c["total_training_rows_across_fits"] or res["ledger_calls"] != c["ledger_paths"]:
        raise InvalidInput("counts differ from spec")
    cmp = compare(res["M2"], anchor)
    check_parents(spec)
    envf_after = check_environment(spec)
    code_sha = sha_full(os.path.abspath(__file__))
    payload = build_payload(spec, dict(before=envf, after=envf_after), res, anchor, cmp, code_sha, datetime.now(timezone.utc))
    if verify_serialized_fits(payload["fits"]) != len(res["models"]):
        raise InvalidInput("serialized fit count differs")
    data = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8")
    if verify_serialized_fits(json.loads(data.decode("utf-8"))["fits"]) != len(res["models"]):
        raise InvalidInput("round-tripped fits differ")
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


def selftest():
    assert fee(10000) == 1 and fee(20000) == 3 and fee(6666) == 0 and fee(6667) == 1
    O = [0, 5000] + [0] * 19 + [5000]
    assert targets(O, 0) == (-10 / 5000, 0.0)
    O[-1] = 5005
    assert targets(O, 0) == (-5 / 5000, 5 / 5000)
    n = 900
    days = synthetic_days(n)
    rng = np.random.default_rng(11)
    Cadj = list(1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n))))
    Oraw = [int(round(c)) * 10 for c in Cadj]
    Craw = [o + 5 * int(rng.integers(-3, 4)) for o in Oraw]
    Oi, Ci, Ca = validate_series(days, Oraw, Craw, Cadj)
    f = features(Ca, 250)
    assert abs(f[0] - math.log(Ca[250] / Ca[245])) < 1e-15 and abs(f[3] - (Ca[250] / (sum(Ca[51:251]) / 200) - 1)) < 1e-12 and features(Ca, FEATURE_START - 1) is None
    r20 = [math.log(Ca[t] / Ca[t - 1]) for t in range(231, 251)]
    assert abs(f[4] - float(np.std(r20))) < 1e-15
    cutoff = 600
    X, y, used = training_rows(Oi, Ca, cutoff)
    assert used[0] == FEATURE_START and used[-1] == cutoff - H - 1 and (cutoff - H) not in used and len(X) == 2 * len(used) and X[0][5] == 0.0 and X[1][5] == 1.0
    FIT_CALLS[0] = 0
    m = fit_lgbm(X, y)
    assert FIT_CALLS[0] == 1 and 1 <= m["num_trees"] <= NUM_ROUNDS and m["reload_parity_max_abs_error"] <= 1e-9 and m["model_string_sha256"] and m["dump_sha256"] and set(m["gain_importance"]) == set(FEATURES)
    dump = m["booster"].dump_model(num_iteration=-1)
    assert sha_bytes(canonical_json_bytes(dump)) == m["dump_sha256"] and m["dump"] == dump and len(m["dump"]["tree_info"]) == m["num_trees"]
    ser = json.loads(json.dumps({"2099": {k: v for k, v in m.items() if k != "booster"}}, allow_nan=False))
    assert verify_serialized_fits(ser) == 1
    broken = json.loads(json.dumps(ser))
    broken["2099"]["dump"]["tree_info"][0]["tree_structure"] = {"leaf_value": 0.0}
    expect_reject(verify_serialized_fits, broken)
    broken = json.loads(json.dumps(ser))
    broken["2099"].pop("dump")
    expect_reject(verify_serialized_fits, broken)
    Xa = np.asarray(X[:50], dtype=np.float64)
    native = m["booster"].predict(Xa, start_iteration=0, num_iteration=-1, num_threads=1)
    manual = np.array([traverse_dump(dump, row) for row in Xa])
    assert np.max(np.abs(native - manual)) <= 1e-9, np.max(np.abs(native - manual))
    reloaded = lgb.Booster(model_str=m["model_string"])
    assert np.max(np.abs(reloaded.predict(Xa, num_threads=1) - native)) <= 1e-9
    p_held0 = predict_model(m, X[0][:5], 0)
    p_held1 = predict_model(m, X[0][:5], 1)
    assert math.isfinite(p_held0) and math.isfinite(p_held1) and abs(p_held0 - native[0]) <= 1e-9 and abs(p_held1 - native[1]) <= 1e-9
    O_p, Ca_p = list(Oi), list(Ca)
    for t in range(cutoff + 1, n):
        O_p[t] += 500
        Ca_p[t] *= 1.3
    X2, y2, used2 = training_rows(O_p, Ca_p, cutoff)
    assert X2 == X and y2 == y and used2 == used
    m2 = fit_lgbm(X2, y2)
    assert m2["model_string_sha256"] == m["model_string_sha256"] and m2["dump_sha256"] == m["dump_sha256"]
    assert training_rows(Oi, Ca, FEATURE_START + H)[2] == [] and training_rows(Oi, Ca, FEATURE_START + H + 1)[2] == [FEATURE_START]
    const = fit_lgbm(X[:600], [0.001] * 600)
    assert const["residual_variance_ratio"] is None and 1 <= const["num_trees"] <= NUM_ROUNDS and abs(predict_model(const, X[0][:5], 0) - 0.001) < 1e-9
    expect_reject(fit_lgbm, X[:1], y[:1])
    expect_reject(fit_lgbm, [[float("nan")] * 6, [0.0] * 6], [0.0, 0.0])
    d5 = synthetic_days(5)
    O5, C5 = [5000] * 5, [5000] * 5
    seq = {0: 1.0, 1: 1.0, 2: 0.5, 3: 1.0}
    led = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="x", w=seq[i]))
    assert [(x["side"], x["qty"], x["price"], x["fee"]) for x in led["trades"]] == [("buy", 1, 5005, 0), ("buy", 2, 5005, 1), ("sell", 1, 4995, 0), ("buy", 5, 5005, 3)]
    assert led["metrics"]["final_cash"] == 14951 and led["metrics"]["qty_end"] == 7 and led["metrics"]["pnl_won_after_costs"] == -49 and led["metrics"]["fees_total_won"] == 4 and led["metrics"]["tick_cost_won"] == 45 and led["metrics"]["execution_opportunities"] == 4
    seqf = {0: 1.0, 1: 0.0, 2: 0.0, 3: 1.0}
    ff = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="x", w=seqf[i]))
    assert [t["kind"] for t in ff["trades"]] == ["entry", "exit", "entry"] and exit_episodes(ff["trades"]) == [dict(exit_session=d5[2], reentry_session=d5[4])] and ff["executions"][2]["side"] is None
    keep = ledger_run(d5, O5, C5, 0, 4, lambda i, held: dict(action="keep", w=float(held)))
    assert keep["metrics"]["trades"] == 0
    hs = lambda pattern: holding_streaks([dict(held=v) for v in pattern])
    assert hs([1, 0, 1, 1, 1]) == dict(count=2, min=1, median=2.0, max=3, mean=2.0)
    assert hs([1, 1, 0, 1, 0, 1, 1, 1]) == dict(count=3, min=1, median=2.0, max=3, mean=2.0)
    assert hs([0, 1, 1, 0]) == dict(count=1, min=2, median=2.0, max=2, mean=2.0)
    assert hs([0, 0]) == dict(count=0, min=None, median=None, max=None, mean=None)
    assert hs([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])["median"] == 2.5
    d6 = synthetic_days(6)
    poor6 = ledger_run(d6, [45000] * 6, [45000] * 6, 0, 5, lambda i, held: dict(action="long", w=1.0))
    assert poor6["metrics"]["trades"] == 1 and poor6["trades"][0]["execution_session"] == d6[5] and poor6["daily"][4]["held"] == 0 and poor6["daily"][5]["held"] == 1
    big_days = synthetic_days(1300, start=date(2011, 10, 1))
    rng2 = np.random.default_rng(5)
    Cb = list(1000.0 * np.exp(np.cumsum(rng2.normal(0.0003, 0.012, 1300))))
    Ob = [int(round(c)) * 10 for c in Cb]
    Cbr = [o + 5 * int(rng2.integers(-3, 4)) for o in Ob]
    Obi, Cbi, Cba = validate_series(big_days, Ob, Cbr, Cb)
    folds = year_folds(big_days, 2013, 2015)
    start_b = folds[0]["fit_cutoff_index"]
    FIT_CALLS[0] = 0
    res = compute(big_days, Obi, Cbi, Cba, start_b, 1299, folds)
    assert res["ledger_calls"] == 1 and res["fit_calls"] == 3 and len(res["models"]) == 3 and res["M2"]["metrics"]["sessions"] == 1300 - start_b and len(res["M2"]["decisions"]) == 1299 - start_b
    for d in res["M2"]["decisions"]:
        assert d["entry_session"][:4] == str(d["fit_entry_year"]) and d["features"]["held"] == d["held_at_close"]
    for d in res["M2"]["decisions"][:40]:
        y = d["fit_entry_year"]
        row = [d["features"][k] for k in FEATURES]
        assert abs(traverse_dump(res["models"][y]["booster"].dump_model(num_iteration=-1), row) - d["predicted_contrast"]) <= 1e-9
    assert sum(res["yearly_M2"].values()) == res["M2"]["metrics"]["pnl_won_after_costs"]
    Ob_f = list(Obi)
    for t in range(folds[1]["fit_cutoff_index"] + 1, 1300):
        Ob_f[t] += 500
    res_f = compute(big_days, Ob_f, Cbi, Cba, start_b, 1299, folds)
    assert res_f["models"][2013]["model_string_sha256"] == res["models"][2013]["model_string_sha256"] and res_f["models"][2014]["model_string_sha256"] == res["models"][2014]["model_string_sha256"] and res_f["models"][2015]["model_string_sha256"] != res["models"][2015]["model_string_sha256"]
    fake_anchor_daily = [dict(session=d["session"], pnl=d["pnl"] + 7, qty=d["qty"]) for d in res["M2"]["daily"]]
    fake_obj = dict(envelope=dict(schema_version="timing-actionvalue-v1", symbol=SYM, artifact_id="anchor_synthetic", spec_sha256="x"), variants={k: dict(metrics=dict(res["M2"]["metrics"], deposits=res["M2"]["metrics"]["deposits"], unique_daily_rows=len(fake_anchor_daily), pnl_won_after_costs=res["M2"]["metrics"]["pnl_won_after_costs"] + 7), daily_records=fake_anchor_daily) for k in ("M1", "R1")})
    anc = anchor_from_obj(fake_obj)
    cmpv = compare(res["M2"], anc)
    assert cmpv["M2_minus_R1"]["role"] == "primary" and cmpv["M2_minus_R1"]["terminal_pnl"] == -7 and cmpv["M2_minus_R1"]["daily_diff_min"] == -7 and cmpv["M2_minus_R1"]["daily_diff_max"] == -7 and sum(cmpv["M2_minus_R1"]["yearly_increments_diff"].values()) == -7
    bad = json.loads(json.dumps(fake_obj))
    bad["variants"]["M1"]["daily_records"].append(dict(bad["variants"]["M1"]["daily_records"][0]))
    expect_reject(anchor_from_obj, bad)
    bad = json.loads(json.dumps(fake_obj))
    bad["envelope"]["symbol"] = "000000"
    expect_reject(anchor_from_obj, bad)
    bad = json.loads(json.dumps(fake_obj))
    bad["_sha256"] = "00" * 32
    expect_reject(anchor_from_obj, bad, expected_sha="11" * 32)
    counts_b = dict(daily_rows_per_ledger=len(fake_anchor_daily), deposits_won_per_ledger=res["M2"]["metrics"]["deposits"], execution_opportunities_per_ledger=res["M2"]["metrics"]["execution_opportunities"])
    assert anchor_calendar_check(anc, big_days, start_b, 1299, counts_b)
    expect_reject(anchor_calendar_check, anc, big_days, start_b + 1, 1299, counts_b)
    d0, d1 = date(2002, 10, 14), date.fromisoformat(CUTOFF)
    span = (d1 - d0).days
    sd_full = [(d0 + timedelta(days=round(k * span / (RETAINED_ROWS - 1)))).isoformat() for k in range(RETAINED_ROWS)] + ["2026-11-06"]
    ok_raw = dict(days=list(sd_full), O=[20000] * FULL_ROWS, C=[20000] * FULL_ROWS)
    ok_adj = dict(days=list(sd_full), C=[20000.0] * FULL_ROWS)
    pd_, pO, pCr, pCa = prepare_prices(ok_raw, ok_adj)
    assert len(pd_) == RETAINED_ROWS and pd_[-1] == CUTOFF
    bad_adj = dict(ok_adj)
    bad_adj["days"] = list(sd_full[:-1]) + ["2026-11-09"]
    expect_reject(prepare_prices, ok_raw, bad_adj)
    expect_reject(prepare_prices, dict(ok_raw, O=ok_raw["O"][:-1]), ok_adj)
    expect_reject(validate_series, ["a", "b"], [100.5, 100], [100, 100], [1.0, 1.0])
    spec = load_spec()
    assert spec["sha256"] == SPEC_SHA256 and check_parents(spec) == 7
    envf = check_environment(spec)
    assert envf["lightgbm"] == LGBM_VERSION
    import tempfile
    tmp = tempfile.NamedTemporaryFile(prefix="timing_lgbm_existing_", suffix=".json", delete=False)
    tmp.write(b"{}")
    tmp.close()
    f0, l0 = FIT_CALLS[0], LEDGER_CALLS[0]
    try:
        expect_reject(run, SPEC, tmp.name)
        assert FIT_CALLS[0] == f0 and LEDGER_CALLS[0] == l0
    finally:
        os.unlink(tmp.name)
    ranchor = load_anchor(spec)
    rdays, rO, rCr, rCa = load_prices(spec)
    rfolds, rstart, rend = fold_facts_check(spec, rdays)
    assert anchor_calendar_check(ranchor, rdays, rstart, rend, spec["counts"]) and ranchor["artifact_id"] == "timing_actionvalue_v1_20261106_122319_f745f401" and len(rfolds) == 12 and rstart == 3031 and rend == 5937 and sum(f_["train_rows"] for f_ in rfolds) == 99940
    assert check_parents(spec) == 7
    print("selftest ok", dict(code_sha256_16=sha_full(os.path.abspath(__file__))[:16], here=os.path.dirname(os.path.abspath(__file__)), spec_sha256_16=spec["sha256"][:16], lightgbm=envf["lightgbm"], real_data="hashes, environment, shapes/calendar, fold counts, anchor metadata only; no real features, labels, fits, predictions or ledgers"))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif not sys.argv[1:]:
        run()
    else:
        print("usage: timing_lgbm.py | selftest")
