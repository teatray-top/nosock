#!/usr/bin/env bash
cd /mnt/g/NoSock/stock/penny/ft
F=fixtures/fillgate_v2
rm -rf $F/out && mkdir -p $F/out/end $F/out/nocal $F/out/nopair
FILLGATE_DATADIR=$F/data/binance FILLGATE_PAIRS=ALGO/USDT FILLGATE_CALENDAR=$F/calendar.json FILLGATE_LOG=$F/out/end/gate_log.json \
  ~/ftenv/bin/python ft_kr_fillgate.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateEnd --fee 0 --cache none --export trades --backtest-directory $F/out/end > $F/out/end/run.log 2>&1
echo "end exit $?"
FILLGATE_DATADIR=$F/data/binance FILLGATE_PAIRS=ALGO/USDT FILLGATE_CALENDAR=$F/calendar_missing.json \
  ~/ftenv/bin/python ft_kr_fillgate.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateEnd --fee 0 --cache none --export none --backtest-directory $F/out/nocal > $F/out/nocal/run.log 2>&1
echo "nocal exit $?"; tail -1 $F/out/nocal/run.log
FILLGATE_DATADIR=$F/data/binance FILLGATE_PAIRS=BTC/USDT FILLGATE_CALENDAR=$F/calendar.json \
  ~/ftenv/bin/python ft_kr_fillgate.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateEnd --fee 0 --cache none --export none --backtest-directory $F/out/nopair > $F/out/nopair/run.log 2>&1
echo "nopair exit $?"; tail -1 $F/out/nopair/run.log
