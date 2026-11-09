#!/usr/bin/env bash
D=~/ftenv/lib/python3.12/site-packages/freqtrade
~/ftenv/bin/freqtrade --version 2>/dev/null | tail -1
for f in optimize/backtesting.py strategy/strategy_helper.py data/dataprovider.py data/history/history_utils.py persistence/trade_model.py; do echo "$f $(sha256sum $D/$f | cut -d' ' -f1)"; done
