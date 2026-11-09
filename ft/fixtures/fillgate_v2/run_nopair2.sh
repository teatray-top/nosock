#!/usr/bin/env bash
cd /mnt/g/NoSock/stock/penny/ft
F=fixtures/fillgate_v2
rm -rf $F/out/nopair2 && mkdir -p $F/out/nopair2
FILLGATE_DATADIR=$F/data/binance FILLGATE_PAIRS=BTC/USDT FILLGATE_CALENDAR=$F/calendar.json \
  ~/ftenv/bin/python ft_kr_fillgate.py backtesting -c $F/config.json --userdir user_data --datadir $F/data/binance --strategy-path $F/strategies --strategy FillGateEnd --fee 0 --cache none --export none --backtest-directory $F/out/nopair2 > $F/out/nopair2/run.log 2>&1
echo "nopair2 exit $?"; grep -m1 "not registered" $F/out/nopair2/run.log | cut -c1-200
