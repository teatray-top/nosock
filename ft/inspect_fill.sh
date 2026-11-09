#!/usr/bin/env bash
F=~/ftenv/lib/python3.12/site-packages/freqtrade/optimize/backtesting.py
grep -n "^HEADERS\|^[A-Z_]*_IDX = \|HEADERS = \[" "$F" | head -20
grep -rn "^HEADERS\|^OPEN_IDX\|^DATE_IDX" ~/ftenv/lib/python3.12/site-packages/freqtrade/optimize/*.py | head
grep -n "_try_close_open_order(" "$F" | head
