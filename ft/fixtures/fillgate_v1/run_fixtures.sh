#!/usr/bin/env bash
cd /mnt/g/NoSock/stock/penny/ft
F=fixtures/fillgate_v1
rm -rf $F/out && mkdir -p $F/out/gated $F/out/control
FILLGATE_DATADIR=$F/data/binance FILLGATE_PAIRS=ALGO/USDT FILLGATE_CALENDAR=$F/calendar.json FILLGATE_LOG=$F/out/gated/gate_log.json \
  ~/ftenv/bin/python ft_kr_fillgate.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateFixture --fee 0 --cache none --export trades --backtest-directory $F/out/gated > $F/out/gated/run.log 2>&1
echo "gated exit $?"
~/ftenv/bin/python ft_kr.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateFixture --fee 0 --cache none --export trades --backtest-directory $F/out/control > $F/out/control/run.log 2>&1
echo "control exit $?"
grep -h "ft_kr" $F/out/gated/run.log $F/out/control/run.log
