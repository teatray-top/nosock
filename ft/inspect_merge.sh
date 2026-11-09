#!/usr/bin/env bash
F=$(~/ftenv/bin/python -c 'import freqtrade.strategy.strategy_helper as s; print(s.__file__)')
sed -n '/^def _prepare_informative_pair/,/^def merge_informative_pair/p' "$F" | head -120
