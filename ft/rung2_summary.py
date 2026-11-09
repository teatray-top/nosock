import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import sweep_summary_v2 as S

DATADIR = os.path.join(HERE, "user_data", "data_kodex", "binance")


def main(root, out_path):
    rows = []
    for w in sorted(os.listdir(root)):
        for strat in sorted(os.listdir(os.path.join(root, w))):
            d = os.path.join(root, w, strat)
            res, z = S.load_result(d)
            st = res["strategy"][strat]
            fp = S.fill_physics(st, DATADIR)
            row = dict(window=w, strategy=strat, timeframe=st["timeframe"], start=st["backtest_start"], end=st["backtest_end"], trades=st["total_trades"], profit_pct=round(100 * st["profit_total"], 2), buy_and_hold_pct=round(100 * st["market_change"], 2), profit_factor=st.get("profit_factor"), max_drawdown_account_pct=round(100 * (st.get("max_drawdown_account") or 0), 2), p_value=st.get("p_value"), winrate=st.get("winrate"), fill_violations=fp["violations"], fill_physics=fp, result_zip=os.path.relpath(z, HERE), result_sha256=S.sha(z))
            row["paper_candidate"] = bool(w == "A" and row["profit_pct"] > 0 and (row["profit_factor"] or 0) > 1.1 and row["trades"] >= 30 and fp["violations"] == 0)
            row["beats_holding"] = row["profit_pct"] > row["buy_and_hold_pct"]
            rows.append(row)
            print(f"{w} {strat:9s} {row['timeframe']:3s} {row['start'][:10]}..{row['end'][:10]} trades {row['trades']:4d} profit {row['profit_pct']:8.2f}% hold {row['buy_and_hold_pct']:8.2f}% pf {round(row['profit_factor'] or 0, 2):5} dd {row['max_drawdown_account_pct']:6.2f}% p {round(row['p_value'], 4) if row['p_value'] is not None else None} viol {row['fill_violations']} paper_candidate {row['paper_candidate']} beats_hold {row['beats_holding']}")
    out = dict(version="rung2-summary-v1", code_sha256=S.sha(os.path.abspath(__file__)), summary_v2_sha256=S.sha(os.path.join(HERE, "sweep_summary_v2.py")), rows=rows)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out_path, S.sha(out_path)[:8])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
