#!/usr/bin/env bash
cd /mnt/g/NoSock/stock/penny/ft
F=fixtures/fillgate_v2
rm -rf $F/out/end && mkdir -p $F/out/end
FILLGATE_DATADIR=$F/data/binance FILLGATE_PAIRS=ALGO/USDT FILLGATE_CALENDAR=$F/calendar.json FILLGATE_LOG=$F/out/end/gate_log.json \
  ~/ftenv/bin/python ft_kr_fillgate.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateEnd --fee 0 --cache none --export trades --backtest-directory $F/out/end > $F/out/end/run.log 2>&1
echo "end exit $?"
