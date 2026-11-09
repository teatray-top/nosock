#!/usr/bin/env bash
ps -eo pid,args | grep -E '[r]un_sweep_v1_1|[f]reqtrade backtesting' || echo "no sweep processes"
F=$(~/ftenv/bin/python -c 'import freqtrade.optimize.backtesting as b; print(b.__file__)')
echo "engine file $F"
sed -n '350,375p' "$F"
grep -n 'fill_up_missing' "$F" ~/ftenv/lib/python3.12/site-packages/freqtrade/data/history/history_utils.py | head
grep -n '^from\|^import' "$F" | grep -i history
