import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pandas as pd

import f1_fill_schedule as F

SCALE = 10000
CAPITAL = 100_000_000
STRATEGY = "NostalgiaForInfinityX8"
FORBIDDEN = ("toss", "cred", "freqtrade", "forward_collect_kr_etf", "forward_history_kr_etf", "forward_runner_kr_etf", "backfill_kr_etf_1m", "dobby_kr")
RESULTS = os.path.join(HERE, "results", "conservative_fill_v1")
PREFLIGHT_DIR = os.path.join(RESULTS, "f1_replay_preflight_v2")
RUN_DIR = os.path.join(RESULTS, "f1_replay_v1")
TS_MIN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00$")
TS_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")
INT_TXT = re.compile(r"^\d+$")


MINUTE_CACHE = {}


class Refused(Exception):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(open(p, "rb").read())


def rel(p):
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(p).replace("\\", "/")


def read_exact(path, want):
    data = open(os.path.join(ROOT, path), "rb").read()
    got = sha_bytes(data)
    if got != want:
        raise Refused(f"hash differs for {path}: {got[:12]} != {want[:12]}")
    return data


def positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def num_txt(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def engine_orders(data):
    with zipfile.ZipFile(io.BytesIO(data)) as f:
        names = [x for x in f.namelist() if x.endswith(".json") and not x.endswith("_config.json")]
        st = json.loads(f.read(names[0]))["strategy"][STRATEGY]
    out, bad = [], []
    for t in st["trades"]:
        for o in t["orders"]:
            k = len(out)
            ts, amt, px = o.get("order_filled_timestamp"), o.get("amount"), o.get("safe_price")
            if not isinstance(ts, int) or isinstance(ts, bool):
                bad.append(f"order {k}: order_filled_timestamp is not an integer ms value")
            if not positive(amt):
                bad.append(f"order {k}: amount not a positive finite number")
            if not positive(px):
                bad.append(f"order {k}: safe_price not a positive finite number")
            entry = bool(o["ft_is_entry"])
            out.append(dict(time=datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if isinstance(ts, int) else None, side="buy" if entry else "sell", amount=amt, price=round(px * SCALE, 6) if positive(px) else px, tag=o.get("ft_order_tag"), is_force_exit=o.get("ft_order_tag") == "force_exit" or (not entry and t.get("exit_reason") == "force_exit" and o is t["orders"][-1])))
    return out, bad


def fillgate_orders(data):
    log = json.loads(data.decode("utf-8"))
    out, bad = [], []
    for k, f in enumerate(log["fills"]):
        ts, amt, px = f.get("bar_utc"), f.get("amount"), f.get("price")
        if not (isinstance(ts, str) and TS_UTC.match(ts)):
            bad.append(f"order {k}: bar_utc is not an explicit +00:00 time")
        if not positive(amt):
            bad.append(f"order {k}: amount not a positive finite number")
        if not positive(px):
            bad.append(f"order {k}: price not a positive finite number")
        out.append(dict(time=pd.Timestamp(ts).to_pydatetime() if isinstance(ts, str) and TS_UTC.match(ts) else None, side="buy" if f["is_entry"] else "sell", amount=amt, price=round(px * SCALE, 6) if positive(px) else px, tag=f.get("tag"), is_force_exit=bool(f.get("is_force_exit"))))
    return out, bad


def ledger_orders(data):
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    out, bad = [], []
    for k, r in enumerate(df.itertuples(index=False)):
        amt, px = num_txt(r.shares), num_txt(r.price_won)
        if not TS_UTC.match(r.time_utc):
            bad.append(f"order {k}: time_utc is not an explicit +00:00 time")
        if amt is None or amt <= 0:
            bad.append(f"order {k}: shares not a positive finite number")
        if px is None or px <= 0:
            bad.append(f"order {k}: price_won not a positive finite number")
        if r.side not in ("buy", "sell"):
            bad.append(f"order {k}: side {r.side!r}")
        out.append(dict(time=pd.Timestamp(r.time_utc).to_pydatetime() if TS_UTC.match(r.time_utc) else None, side=r.side, amount=amt, price=px, tag=None, is_force_exit=False))
    return out, bad


READERS = {"engine_zip": engine_orders, "fillgate_log": fillgate_orders, "ledger_csv": ledger_orders}


def to_f1(kind, src_sha, raw):
    out = []
    for i, o in enumerate(raw):
        n = o["amount"]
        shares = int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) and float(n).is_integer() else n
        out.append(dict(order_id=f"{kind}:{src_sha[:12]}:{i}", source_index=i, side=o["side"], shares=shares, activation=o["time"], price=o["price"], tag=o["tag"], is_force_exit=o["is_force_exit"]))
    return out


def minute_rows(data, first_day, last_day):
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    rows, bad = {}, []
    for k, r in enumerate(df.itertuples(index=False)):
        if not (first_day <= r.ts[:10] <= last_day):
            continue
        p = []
        if not TS_MIN.match(r.ts):
            p.append("ts is not a naive KST minute label")
        vals = {c: num_txt(getattr(r, c)) for c in ("open", "high", "low", "close")}
        if any(v is None or v <= 0 for v in vals.values()):
            p.append("price not a positive finite number")
        if not INT_TXT.match(r.volume):
            p.append("volume not a non-negative integer literal")
        if p:
            bad.append(f"row {k} {r.ts}: " + ", ".join(p))
            continue
        t = datetime.fromisoformat(r.ts).replace(tzinfo=F.KST)
        rows.setdefault(r.ts[:10], []).append(dict(label=t, volume=int(r.volume), **vals))
    return rows, bad


def baseline_orders(name):
    import sensitivity_apply as SA
    won, _ = SA.load_orders(SA.SERIES[name])
    return [(o["time"].isoformat(), o["side"], o["shares"], o["price"]) for o in won]


def project_modules():
    return sorted(n for n, m in sys.modules.items() if getattr(m, "__file__", None) and os.path.abspath(m.__file__).startswith(ROOT))


def verify(spec_path, spec_sha, baseline=baseline_orders):
    spec_bytes = open(spec_path, "rb").read()
    if sha_bytes(spec_bytes) != spec_sha:
        raise Refused("spec hash differs")
    spec = json.loads(spec_bytes.decode("utf-8"))
    me = sha_file(os.path.abspath(__file__))
    if me != spec["replay_code"]["sha256"]:
        raise Refused("replay code hash differs from the spec")
    if sha_file(F.__file__) != spec["frozen_inputs"]["f1_code"]["sha256"]:
        raise Refused("f1 code hash differs from the spec")
    files = []
    for k, v in spec["frozen_inputs"].items():
        for item in (v.values() if "file" not in v else [v]):
            read_exact(item["file"], item["sha256"])
            files.append(dict(role=k, file=item["file"], sha256=item["sha256"]))
    cal_bytes = read_exact(spec["frozen_inputs"]["calendar"]["file"], spec["frozen_inputs"]["calendar"]["sha256"])
    cal = json.loads(cal_bytes.decode("utf-8"))["days"]
    raw_bytes = {}
    for name, t in spec["targets"].items():
        for role in ("order_source", "minute_input", "baseline_reference"):
            item = t[role]
            if item["sha256"] not in raw_bytes:
                raw_bytes[item["sha256"]] = read_exact(item["file"], item["sha256"])
            files.append(dict(role=f"{name}.{role}", file=item["file"], sha256=item["sha256"]))
    parsed, report = {}, {}
    minute_cache = MINUTE_CACHE
    for name, t in spec["targets"].items():
        raw, order_bad = READERS[t["kind"]](raw_bytes[t["order_source"]["sha256"]])
        orders = to_f1(t["kind"], t["order_source"]["sha256"], raw)
        r = t["range"]
        days = {d: cal[d] for d in cal if r["first_session"] <= d <= r["last_session"]}
        key = (t["minute_input"]["sha256"], r["first_session"], r["last_session"])
        if key not in minute_cache:
            minute_cache[key] = minute_rows(raw_bytes[t["minute_input"]["sha256"]], r["first_session"], r["last_session"])
        minutes, minute_bad = minute_cache[key]
        mine = [(o["activation"].isoformat() if o["activation"] else None, o["side"], o["shares"], o["price"]) for o in orders]
        base = baseline(name)
        acct = json.loads(raw_bytes[t["baseline_reference"]["sha256"]].decode("utf-8"))["accounting"]
        acct_seq = sorted((a["time"], a["side"], a["requested"]) for a in acct)
        mine_seq = sorted((str(pd.Timestamp(o["activation"])) if o["activation"] else "", o["side"], o["shares"]) for o in orders)
        first_diff = next((i for i, (a, b) in enumerate(zip(mine, base)) if a != b), None if len(mine) == len(base) else min(len(mine), len(base)))
        problems = F.precheck(sorted(days), minutes, days) if not minute_bad else {}
        order_problems = F.check_orders(orders) if not order_bad else []
        non_int = sum(1 for o in orders if not isinstance(o["shares"], int))
        start_active = sum(1 for o in orders if o["activation"] and o["activation"].astimezone(F.KST).strftime("%H:%M") == days.get(o["activation"].astimezone(F.KST).date().isoformat(), {}).get("session_start"))
        ok = first_diff is None and mine_seq == acct_seq and not minute_bad and not problems and not order_bad and not order_problems and non_int == 0
        report[name] = dict(ok=ok, orders=len(orders), equal_to_baseline_order_list=first_diff is None, first_difference=first_diff, equal_to_baseline_accounting=mine_seq == acct_seq, sessions=len(days), minutes=sum(len(v) for v in minutes.values()), minute_raw_problems=minute_bad[:20], minute_raw_problem_count=len(minute_bad), minute_problems=problems, order_raw_problems=order_bad[:20], order_problems=order_problems, non_integer_orders=non_int, session_start_activations=start_active, price_unit=dict(engine_zip="safe_price x 10000", fillgate_log="price x 10000", ledger_csv="price_won as is")[t["kind"]])
        parsed[name] = dict(orders=orders, days=days, minutes=minutes)
    forbidden = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN)
    all_ok = all(v["ok"] for v in report.values()) and not forbidden
    return dict(spec=spec, spec_bytes=spec_bytes, replay_sha=me, files=files, report=report, parsed=parsed, all_ok=all_ok, forbidden=forbidden)


def summarize(name, res, cons, orders, base):
    if res["status"] != "ok":
        return dict(status=res["status"], problems=res["problems"], baseline_return_pct=base["return_pct"])
    if not cons["ok"]:
        return dict(status="conservation_failed", conservation=cons, baseline_return_pct=base["return_pct"])
    by_id = {o["order_id"]: o for o in orders}
    diffs = [dict(order_id=fl["order_id"], side=fl["side"], won=fl["price"] - by_id[fl["order_id"]]["price"], bps=1e4 * (fl["price"] / by_id[fl["order_id"]]["price"] - 1), delay_min=(datetime.fromisoformat(fl["model_fill_time"]) - by_id[fl["order_id"]]["activation"]).total_seconds() / 60) for fl in res["fills"]]
    statuses = {}
    for r in res["orders"]:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    ret = 100 * (res["final"]["net_liquidation_value"] / CAPITAL - 1)
    return dict(status="ok", f1_nlv_won=round(res["final"]["net_liquidation_value"]), f1_return_pct=round(ret, 4), baseline_return_pct=base["return_pct"], f1_minus_baseline_pct=round(ret - base["return_pct"], 4), min_cash_won=round(res["final"]["min_cash"], 4), executable=res["final"]["executable"], order_status_counts=statuses, conservation_ok=True, fill_price_diff=diffs, counters=res["counters"])


def preflight_doc(v, spec_path, spec_sha, out_file):
    return dict(version="f1-replay-preflight-v2", spec=dict(file=rel(spec_path), sha256=spec_sha), replay_code=dict(file=rel(os.path.abspath(__file__)), sha256=v["replay_sha"]), f1_code=dict(file=rel(F.__file__), sha256=sha_file(F.__file__)), files=v["files"], targets=v["report"], all_ok=v["all_ok"], imports=dict(project_modules=project_modules(), forbidden_loaded=v["forbidden"]), write_paths=[rel(out_file)], no_f1_run=True)


def preflight(spec_path, spec_sha, out_dir=PREFLIGHT_DIR, extra=None):
    v = verify(spec_path, spec_sha)
    out_file = os.path.join(out_dir, "preflight.json")
    doc = preflight_doc(v, spec_path, spec_sha, out_file)
    if extra:
        doc["failure_path_checks"] = extra
        doc["write_paths"].append("temporary directory under the system temp folder for the failure-path checks, removed afterwards")
    doc["run_meta"] = dict(generated_at=datetime.now(F.KST).isoformat(timespec="seconds"), python=platform.python_version(), dont_write_bytecode=sys.dont_write_bytecode, excluded_from_review_hash=True)
    os.makedirs(out_dir, exist_ok=False)
    data = json.dumps(doc, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    with open(out_file, "xb") as f:
        f.write(data)
    return doc, sha_bytes(data)


def gate(v, preflight_path, preflight_sha):
    data = open(preflight_path, "rb").read()
    if sha_bytes(data) != preflight_sha:
        raise Refused("preflight file hash differs")
    pf = json.loads(data.decode("utf-8"))
    if not pf.get("all_ok"):
        raise Refused("preflight is not all_ok")
    if pf["spec"]["sha256"] != sha_bytes(v["spec_bytes"]) or pf["replay_code"]["sha256"] != v["replay_sha"]:
        raise Refused("preflight was made for another spec or runner")
    if not v["all_ok"]:
        bad = [k for k, r in v["report"].items() if not r["ok"]]
        raise Refused(f"verification failed before any F1 run: {bad or v['forbidden']}")


def run(spec_path, spec_sha, preflight_path, preflight_sha, out_dir=RUN_DIR, f1_run=None, baseline=baseline_orders):
    f1_run = f1_run or F.run
    v = verify(spec_path, spec_sha, baseline=baseline)
    gate(v, preflight_path, preflight_sha)
    spec = v["spec"]
    os.makedirs(out_dir, exist_ok=False)
    with open(os.path.join(out_dir, "spec_copy.json"), "xb") as f:
        f.write(v["spec_bytes"])
    with open(os.path.join(out_dir, "inputs.json"), "x", encoding="utf-8") as f:
        json.dump(dict(spec=dict(file=rel(spec_path), sha256=spec_sha), preflight=dict(file=rel(preflight_path), sha256=preflight_sha), replay_code=v["replay_sha"], f1_code=sha_file(F.__file__), files=v["files"]), f, indent=1)
    summary = dict(version="f1-replay-v1", label=spec["name"] + "; " + spec["what_it_is_not"], capital_won=CAPITAL, targets={})
    for name in spec["targets"]:
        p = v["parsed"][name]
        t = spec["targets"][name]
        res = f1_run(p["orders"], p["minutes"], p["days"], t["range"]["first_session"], t["range"]["last_session"], CAPITAL)
        cons = F.conservation(res, CAPITAL)
        with open(os.path.join(out_dir, f"{name}.json"), "x", encoding="utf-8") as f:
            json.dump(dict(target=name, result=res, conservation=cons), f, ensure_ascii=False, indent=1, default=str)
        summary["targets"][name] = summarize(name, res, cons, p["orders"], t["baseline_reference"])
    summary["imports"] = dict(project_modules=project_modules(), forbidden_loaded=sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN))
    code_dir = os.path.join(out_dir, "code")
    os.makedirs(code_dir)
    for src in (os.path.abspath(__file__), F.__file__):
        shutil.copyfile(src, os.path.join(code_dir, os.path.basename(src)))
    data = json.dumps(summary, ensure_ascii=False, indent=1, default=str).encode("utf-8")
    with open(os.path.join(out_dir, "summary.json"), "xb") as f:
        f.write(data)
    return summary, sha_bytes(data)


def failure_checks(spec_path, spec_sha, good_preflight):
    spec = json.loads(open(spec_path, "rb").read().decode("utf-8"))
    names = list(spec["targets"])
    last = names[-1]
    tmp = tempfile.mkdtemp(prefix="f1_replay_fail_")
    out = []
    try:
        def write_spec(s, tag):
            p = os.path.join(tmp, f"spec_{tag}.json")
            b = json.dumps(s, ensure_ascii=False, indent=1).encode("utf-8")
            open(p, "wb").write(b)
            return p, sha_bytes(b)

        def copy_with(src_rel, tag, edit):
            data = open(os.path.join(ROOT, src_rel), "rb").read()
            new = edit(data)
            p = os.path.join(tmp, tag + os.path.splitext(src_rel)[1])
            open(p, "wb").write(new)
            return p, sha_bytes(new)

        def attempt(tag, s_path, s_sha, pf, expect, baseline=baseline_orders):
            calls = []
            run_dir = os.path.join(tmp, f"run_{tag}")

            def counting(*a, **k):
                calls.append(1)
                return F.run(*a, **k)
            try:
                run(s_path, s_sha, pf[0], pf[1], out_dir=run_dir, f1_run=counting, baseline=baseline)
                outcome = "ran"
            except Refused as e:
                outcome = f"refused: {e}"
            out.append(dict(case=tag, outcome=outcome[:300], f1_run_calls=len(calls), run_dir_created=os.path.exists(run_dir), passed=outcome.startswith("refused") and expect in outcome and not calls and not os.path.exists(run_dir)))

        def fresh_preflight(s_path, s_sha, tag):
            d = os.path.join(tmp, f"pf_{tag}")
            try:
                _, h = preflight(s_path, s_sha, out_dir=d)
            except Refused as e:
                return None, str(e)
            return (os.path.join(d, "preflight.json"), h), None

        s = json.loads(json.dumps(spec))
        s["targets"][last]["minute_input"]["sha256"] = "0" * 64
        p, h = write_spec(s, "last_minute_hash")
        attempt("last target minute hash differs", p, h, good_preflight, "hash differs")

        s = json.loads(json.dumps(spec))
        s["targets"][last]["baseline_reference"]["sha256"] = "1" * 64
        p, h = write_spec(s, "last_baseline_hash")
        attempt("last target baseline reference hash differs", p, h, good_preflight, "hash differs")

        other = spec["targets"][names[-2]]["order_source"]
        s = json.loads(json.dumps(spec))
        s["targets"][last]["order_source"] = dict(other)
        p, h = write_spec(s, "last_mapping")
        attempt("last target order source swapped (mapping differs from the baseline input)", p, h, good_preflight, "preflight was made for another spec")
        pf, err = fresh_preflight(p, h, "mapping")
        if pf:
            attempt("same, with its own fresh preflight", p, h, pf, "preflight is not all_ok")
        attempt("in-process mapping differs although the preflight passed", spec_path, spec_sha, good_preflight, "verification failed before any F1 run", baseline=lambda n: baseline_orders(n)[:-1] if n == last else baseline_orders(n))

        base_min = spec["targets"][last]["minute_input"]["file"]

        def volume_09(data):
            lines = data.decode("utf-8").split("\n")
            day = spec["targets"][last]["range"]["first_session"]
            i = next(i for i, l in enumerate(lines) if l.startswith(day + "T09:31"))
            parts = lines[i].split(",")
            parts[-1] = "0.9"
            lines[i] = ",".join(parts)
            return "\n".join(lines).encode("utf-8")

        def ts_offset(data):
            day = spec["targets"][last]["range"]["first_session"]
            return data.replace(f"{day}T09:31:00,".encode(), f"{day}T09:31:00+00:00,".encode(), 1)
        for tag, edit, expect in (("minute volume 0.9", volume_09, "preflight is not all_ok"), ("minute ts with +00:00", ts_offset, "preflight is not all_ok")):
            mp, mh = copy_with(base_min, tag.replace(" ", "_").replace("+", "p").replace(":", ""), edit)
            s = json.loads(json.dumps(spec))
            s["targets"][last]["minute_input"] = dict(file=mp, sha256=mh)
            p, h = write_spec(s, tag.replace(" ", "_").replace("+", "p").replace(":", ""))
            pf, err = fresh_preflight(p, h, tag.replace(" ", "_").replace("+", "p").replace(":", ""))
            pdoc = json.load(open(pf[0], encoding="utf-8")) if pf else {}
            raw_problems = pdoc.get("targets", {}).get(last, {}).get("minute_raw_problems", [])
            attempt(tag + " (no silent coercion)", p, h, pf, expect)
            out[-1]["preflight_raw_problems"] = raw_problems[:3]
            out[-1]["passed"] = out[-1]["passed"] and bool(raw_problems)

        ledger_name = next(n for n in names if spec["targets"][n]["kind"] == "ledger_csv")
        src = spec["targets"][ledger_name]["order_source"]["file"]

        def shares_eps(data):
            lines = data.decode("utf-8").split("\n")
            parts = lines[1].split(",")
            parts[2] = parts[2] + ".0000000001"
            lines[1] = ",".join(parts)
            return "\n".join(lines).encode("utf-8")
        lp, lh = copy_with(src, "ledger_shares_eps", shares_eps)
        s = json.loads(json.dumps(spec))
        s["targets"][ledger_name]["order_source"] = dict(file=lp, sha256=lh)
        p, h = write_spec(s, "shares_eps")
        pf, err = fresh_preflight(p, h, "shares_eps")
        pdoc = json.load(open(pf[0], encoding="utf-8")) if pf else {}
        attempt("ledger shares 1.0000000001 style (not rounded)", p, h, pf, "preflight is not all_ok")
        out[-1]["preflight_non_integer_orders"] = pdoc.get("targets", {}).get(ledger_name, {}).get("non_integer_orders")
        out[-1]["passed"] = out[-1]["passed"] and out[-1]["preflight_non_integer_orders"] == 1

        attempt("no valid preflight file (hash differs)", spec_path, spec_sha, (good_preflight[0], "2" * 64), "preflight file hash differs")

        fake = dict(status="ok", problems={}, fills=[], orders=[], counters={}, final=dict(net_liquidation_value=CAPITAL, min_cash=CAPITAL, executable=True))
        sm = summarize("x", fake, dict(ok=False, per_order=False), [], dict(return_pct=1.0))
        out.append(dict(case="conservation failure is not reported as ok", outcome=sm["status"], f1_run_calls=0, run_dir_created=False, passed=sm["status"] == "conservation_failed" and "f1_return_pct" not in sm))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["preflight", "run"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--spec-sha", required=True)
    ap.add_argument("--preflight")
    ap.add_argument("--preflight-sha")
    a = ap.parse_args()
    spec_path = os.path.abspath(a.spec)
    try:
        if a.cmd == "preflight":
            tmp = tempfile.mkdtemp(prefix="f1_replay_pf_")
            try:
                _, h0 = preflight(spec_path, a.spec_sha, out_dir=os.path.join(tmp, "pf"))
                checks = failure_checks(spec_path, a.spec_sha, (os.path.join(tmp, "pf", "preflight.json"), h0))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            doc, h = preflight(spec_path, a.spec_sha, extra=checks)
            for k, r in doc["targets"].items():
                print(k, "ok", r["ok"], "orders", r["orders"], "baseline_list", r["equal_to_baseline_order_list"], "accounting", r["equal_to_baseline_accounting"], "minutes", r["minutes"], "start_active", r["session_start_activations"])
            for c in checks:
                print("PASS" if c["passed"] else "FAIL", c["case"], "|", c["outcome"][:120], "| f1 calls", c["f1_run_calls"], "| run dir", c["run_dir_created"])
            print("all_ok", doc["all_ok"], "failure checks passed", all(c["passed"] for c in checks), "preflight", h)
        else:
            summary, h = run(spec_path, a.spec_sha, os.path.abspath(a.preflight), a.preflight_sha)
            for k, v in summary["targets"].items():
                print(k, v.get("status"), v.get("f1_return_pct"), "baseline", v.get("baseline_return_pct"), "diff", v.get("f1_minus_baseline_pct"))
            print("summary", h)
    except Refused as e:
        raise SystemExit(f"refused: {e}")


if __name__ == "__main__":
    main()
