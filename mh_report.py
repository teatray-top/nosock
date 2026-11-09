import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def pct(x, nd=2):
    return "n/a" if x is None else f"{x * 100:+.{nd}f}%"


def won(x):
    return "n/a" if x is None else f"{x:+,.0f}원"


def head_table(rows):
    out = ["| 폴드 | 상태 | 학습 n | 검증 성숙 n | τ(분위) | 시험 신호 | 적격 신호 n / 평균 / 중앙 / 승률 | 적격 모집단 n / 평균 | 부적격 행 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ch = r.get("chosen")
        t = r.get("test", {})
        sg = t.get("signals_eligible_stats", {}) or {}
        pop = t.get("population_eligible", {}) or {}
        tau = f"{ch['tau']:.5f} (q{ch['q']})" if ch else "현금"
        out.append(f"| {r['fold']} | {r['status']} | {r.get('n_train', 0):,} | {r.get('n_val_mature', 0):,} | {tau} | {t.get('signals', 0)} | {sg.get('n', 0)} / {pct(sg.get('mean'))} / {pct(sg.get('median'))} / {pct(sg.get('hit'), 0)} | {pop.get('n', 0):,} / {pct(pop.get('mean'))} | {t.get('n_test_ineligible', 0)} |")
    return out


def ledger_table(reports):
    out = ["| 원장 | 입금 | 확정 조정손익 | 손익 가용 종료일 | 불가 시점 | 가정 시나리오(불확실 평가 포함) | 불가 전 낙폭(원) | 불가 전 성과 낙폭% | 거래 | 미체결 | 지연 | 이월 | 불확실(평가액) | 경로불명 이월 | 중단일 | 자료문제일 | 미청산 | 상위3일 제외 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in reports.items():
        c = r["counts"]
        unavail = r.get("pnl_unavailable_from") or "-"
        through = r.get("pnl_available_through") or "-"
        out.append(f"| {name} | {r['deposits']:,.0f} | {won(r.get('adj_pnl_confirmed'))} | {through} | {unavail} | {won(r.get('adj_pnl_incl_uncertain_marks_scenario'))} | {won(r.get('max_dd_won'))} | {pct(r.get('max_dd_perf_pct_of_deposits'), 1)} | {r['n_trades']} | {c.get('S_unfilled', 0)} | {c.get('S_delayed', 0)} | {c.get('S_carried', 0)} | {c.get('S_exit_unknown_open', 0)} ({r['uncertain_open_value']:,.0f}) | {c.get('S_carried_path_unknown', 0)} | {c.get('S_stop_days', 0)} | {c.get('S_days_with_data_issues', 0)} | {len(r['open_positions'])} | {won(r.get('adj_pnl_ex_top3'))} |")
    return out


def build(train_dir):
    res = json.load(open(os.path.join(train_dir, "results.json"), encoding="utf-8"))
    meta = json.load(open(os.path.join(train_dir, "meta.json"), encoding="utf-8"))
    lpath = os.path.join(train_dir, "ledger", "ledger_results.json")
    led = json.load(open(lpath, encoding="utf-8")) if os.path.exists(lpath) else None
    title = "중장기(L) 단독 / 5원장 결과 보고" if not res.get("S") else "통합 모델 v1 결과 보고"
    lines = [f"# {title} (생성 {datetime.now().isoformat(timespec='minutes')})", "", "탐색적 과거 진단. 제공처 조정/명목 가격·가정 비용(수수료 0.015%×2, 틱 5원 양쪽)·정수 주 기준. 통계 판정 없음. 매수 권고 아님. 주문 없음.", "", f"- 학습 코드 {meta.get('script_sha256')}, 테이블 {meta.get('tables_sha256')}, 적합 {res.get('budget', {}).get('fits')}회 / {res.get('budget', {}).get('wall_s')}초.", ""]
    for head, title in (("S", "헤드 S (15분 보유, 당일 종료)"), ("L", "헤드 L (60거래일 보유)")):
        lines += [f"## {title}", ""] + head_table(res.get(head, [])) + [""]
    if led:
        lines += ["## 세 원장 비교 (공통 구간 " + " ~ ".join(led["meta"]["window"]) + f", 세션 {led['meta']['sessions']}일, 세션당 입금 {led['meta']['deposit']:,.0f}원, 통합 분배 S {led['meta']['shares']['combined'][0]:.0%} / L {led['meta']['shares']['combined'][1]:.0%})", ""] + ledger_table(led["reports"]) + ["", "확정 조정손익 = 확정 총자산(현금+금고+확실한 포지션 평가) − 누적 입금. 청산이 결측으로 불명인 포지션이 생기면 그 날부터 주원장 손익·낙폭은 **불가(n/a)**로 두고 '불가 시점'에 날짜를 적는다. '가정 시나리오' 열은 불확실 포지션을 보유 중인 것으로 평가한 가정값이고, `*_unknown_nextday_scenario` 원장은 익일 청산 가정이다. 낙폭은 불가 전 구간의 확정 계열 기준, 성과 낙폭%는 당시 누적 입금 대비.", ""]
    if led and led.get("bear_windows"):
        rule = led["meta"].get("bear_rule", {})
        lines += [f"## 하락장 창 평가 (규칙: {rule.get('sym')} 조정 종가가 직전 {rule.get('lookback')}세션 고점 대비 {rule.get('drawdown', 0)*100:.0f}% 이상 하락한 첫날 ~ 그 고점 회복일, 미회복이면 자료 끝; 각 창 시작일부터 같은 입금)", ""]
        hs = [str(h) for h in rule.get("horizons", [])]
        lines += ["| 창(확인일~회복일) | 회복 | 세션 | 저점 | 모델 지원 | 원장 | 끝 손익 / 창 입금 | " + " | ".join(f"+{h}세션 손익" for h in hs) + " | 끝까지 낙폭(원) | 끝까지 청산 거래 |", "|---|---|---|---|---|---|---|" + "---|" * len(hs) + "---|---|"]
        for w in led["bear_windows"]:
            win = w["window"]
            for name, m in w["ledgers"].items():
                hz = " | ".join((won(m["horizons"][h]["adj_pnl"]) if m["horizons"].get(h) else "n/a") for h in hs)
                dep = f"{m['deposits_at_end']:,.0f}" if m.get("deposits_at_end") is not None else "n/a"
                lines.append(f"| {win['start']}~{win['end']} | {'예' if win['recovered'] else '아니오(미완료)'} | {win['sessions']} | {pct(win['trough']/win['peak']-1, 1)} | {w.get('model_support', '있음' if w['model_available'] else '없음')} | {name} | {won(m['adj_pnl_at_end'])} / {dep} | {hz} | {won(m.get('max_dd_won_pre_unavailable'))} | {m['n_trades'] if m['n_trades'] is not None else 'n/a'} |")
        lines += ["", "창 시작 = 20% 하락이 종가로 확인된 날, 진입은 그 다음 세션 시가부터(확인일 결정만 사용). 창 시작 시점의 입금 0원에서 출발하는 적립 원장이라 하락 전 보유자의 손실과 다르고 일시 투자 수익률도 아니다. 모델 원장은 창 시작부터 예측이 완전히 지원되는 창(시험 폴드 2015~2026)에만 있고, 2015년 이전 창은 '모델 미지원'(현금으로 대체하지 않음), 부분 지원 창은 실행하지 않음. 미회복 창은 미완료로 표시하며 회복한 창만 모아 집계하지 않는다. 각 열은 해당 시점까지의 지표(끝 뒤 연장 구간 제외).", ""]
    if led and led.get("yearly"):
        lines += ["## 연도별 원장 (각 연도 첫 세션에 현금 0으로 시작, 세션당 입금 동일; 하위 구간은 연도 원장 포지션을 이어받고 구간 시작 순자산·구간 입금을 분리)", ""]
        lines += ["| 연도 | 모델 | 원장 | 연간 손익 / 입금 | 연간 낙폭(원) | 평균 노출 | 거래 | 하위 구간 | 구간 시작 순자산 | 구간 입금 | 구간 손익 | 구간 낙폭(원) | 구간 노출 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for yr in led["yearly"]:
            for name, m in yr["ledgers"].items():
                fy = m["full_year"]
                subs = m.get("sub_periods") or {}
                if not subs:
                    subs = {"-": None}
                for key, sp in subs.items():
                    sp_cells = f"{key} | {sp['start_total']:,.0f} | {sp['deposits']:,.0f} | {won(sp['pnl'])} | {won(sp['max_dd_won'])} | {pct(sp['exposure_avg'], 0)}" if sp else "- | - | - | - | - | -"
                    lines.append(f"| {yr['year']} | {'있음' if yr['model_available'] else '없음'} | {name} | {won(fy['pnl'])} / {fy['deposits']:,.0f} | {won(fy['max_dd_won'])} | {pct(fy['exposure_avg'], 0)} | {m['n_trades']} | {sp_cells} |")
        lines += [""]
    lines += ["## 읽을 때 주의", "", "- 시험 구간은 과거 추가 검증이며 미래 성과가 아니다. 1분봉 구간 전체가 한 번의 설계로 본 결과라 재학습·재선택은 없다.", "- 체결은 다음 봉 시가 ± 1틱 가정이며 실제 호가·지연·부분체결은 미반영(운영 점검 대상).", "- 미체결·지연·이월·불확실 건수는 체결 가능성의 한계를 보여 주는 숫자이며 성과에 포함되지 않은 가정의 크기다.", ""]
    return "\n".join(lines)


def main(train_dir):
    text = build(train_dir)
    out = os.path.join(train_dir, "report.md")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("saved", out, "report sha", sha(out), "script sha", sha(os.path.abspath(__file__)))


def selftest():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "ledger"))
    json.dump(dict(script_sha256="abc", tables_sha256="def"), open(os.path.join(d, "meta.json"), "w", encoding="utf-8"))
    fold = dict(fold="F1", status="fitted", n_train=1000, n_val_mature=200, chosen=dict(tau=0.0012, q=0.99), test=dict(signals=10, signals_eligible=9, signals_eligible_stats=dict(n=9, mean=0.001, median=0.0005, hit=0.56), population_eligible=dict(n=500, mean=-0.0005), n_test_ineligible=3))
    json.dump(dict(S=[fold, dict(fold="F2", status="fitted", n_train=1, n_val_mature=1, chosen=None, test=dict(signals=0, signals_eligible_stats={}, population_eligible=dict(n=0), n_test_ineligible=0))], L=[dict(fold, fold="L2024")], budget=dict(fits=20, wall_s=12.3)), open(os.path.join(d, "results.json"), "w", encoding="utf-8"))
    rep = dict(deposits=100000.0, adj_pnl_confirmed=None, pnl_available_through="2025-03-03", pnl_unavailable_from="2025-03-04", adj_pnl_incl_uncertain_marks_scenario=80.0, max_dd_won=-300.0, max_dd_perf_pct_of_deposits=-0.01, n_trades=4, counts=dict(S_unfilled=1, S_delayed=0, S_carried=1, S_exit_unknown_open=1, S_carried_path_unknown=1, S_stop_days=0, S_days_with_data_issues=2), uncertain_open_value=200.0, open_positions=[{}], adj_pnl_ex_top3=None)
    bw = [dict(window=dict(start="2020-03-12", end="2020-07-20", recovered=True, peak=100.0, trough=66.0, sessions=88), run_end="2021-03-15", model_available=True, model_support="full", ledgers=dict(long_only_model=dict(deposits_at_end=880000.0, adj_pnl_at_end=12345.0, total_at_end=892345.0, pnl_available_at_end=True, horizons={"60": dict(day="2020-06-09", adj_pnl=5000.0, deposits=600000.0), "120": None, "250": dict(day="2021-03-15", adj_pnl=20000.0, deposits=2500000.0)}, n_trades=3, max_dd_won_pre_unavailable=-4000.0), cash=dict(deposits_at_end=880000.0, adj_pnl_at_end=0.0, total_at_end=880000.0, pnl_available_at_end=True, horizons={"60": dict(day="2020-06-09", adj_pnl=0.0, deposits=600000.0), "120": None, "250": None}, n_trades=0, max_dd_won_pre_unavailable=0.0)))]
    pm = dict(start="2020-01-02", end="2020-12-30", sessions=248, start_total=0.0, deposits=2480000.0, end_total=2500000.0, pnl=20000.0, pnl_available=True, max_dd_won=-30000.0, exposure_avg=0.55, exposure_max=0.9, open_marked_end=100.0, uncertain_marked_end=0.0)
    sp = dict(pm, start="2020-02-20", end="2020-03-23", sessions=23, start_total=300000.0, deposits=230000.0, pnl=-45000.0, max_dd_won=-50000.0, exposure_avg=0.7)
    yearly = [dict(year="2020", sessions=248, model_available=True, ledgers=dict(long_only_model=dict(full_year=pm, n_trades=4, sub_periods={"2020-02-20~2020-03-23": sp}), cash=dict(full_year=dict(pm, pnl=0.0, max_dd_won=0.0, exposure_avg=0.0), n_trades=0, sub_periods={})))]
    json.dump(dict(meta=dict(window=["2024-04-01", "2026-11-05"], sessions=600, deposit=10000.0, shares=dict(combined=[0.3, 0.7]), bear_rule=dict(sym="069500", drawdown=0.2, lookback=250, horizons=[60, 120, 250])), reports=dict(combined_model=rep, cash=dict(rep, adj_pnl_confirmed=0.0, pnl_available_through="2026-11-05", pnl_unavailable_from=None, adj_pnl_incl_uncertain_marks_scenario=0.0, n_trades=0, adj_pnl_ex_top3=0.0)), bear_windows=bw, yearly=yearly), open(os.path.join(d, "ledger", "ledger_results.json"), "w", encoding="utf-8"))
    text = build(d)
    assert "| 2020-03-12~2020-07-20 | 예 | 88 | -34.0% | full | long_only_model | +12,345원 / 880,000 | +5,000원 | n/a | +20,000원 | -4,000원 | 3 |" in text, text
    assert "| 2020 | 있음 | long_only_model | +20,000원 / 2,480,000 | -30,000원 | +55% | 4 | 2020-02-20~2020-03-23 | 300,000 | 230,000 | -45,000원 | -50,000원 | +70% |" in text, text
    assert "| 2020 | 있음 | cash | +0원 / 2,480,000 | +0원 | +0% | 0 | - | - | - | - | - | - |" in text, text
    assert text.startswith("# 통합 모델 v1 결과 보고")
    res2 = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))
    res2["S"] = []
    json.dump(res2, open(os.path.join(d, "results.json"), "w", encoding="utf-8"))
    assert build(d).startswith("# 중장기(L) 단독 / 5원장 결과 보고")
    assert "| F1 | fitted | 1,000 | 200 | 0.00120 (q0.99) | 10 | 9 / +0.10% / +0.05% / +56% | 500 / -0.05% | 3 |" in text, text
    assert "| F2 | fitted | 1 | 1 | 현금 | 0 |" in text
    assert "| combined_model | 100,000 | n/a | 2025-03-03 | 2025-03-04 | +80원 | -300원 | -1.0% | 4 | 1 | 0 | 1 | 1 (200) | 1 | 0 | 2 | 1 | n/a |" in text, text
    assert "| cash | 100,000 | +0원 | 2026-11-05 | - | +0원 |" in text, text
    assert "불가 전 낙폭" in text
    assert "매수 권고 아님" in text and "통계 판정 없음" in text
    print("selftest ok", dict(sha=sha(os.path.abspath(__file__))))


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("usage: mh_report.py <train_dir> | selftest")
