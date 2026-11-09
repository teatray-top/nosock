#!/usr/bin/env bash
set -u
cd /mnt/g/NoSock/stock/penny/ft
FT=~/ftenv/bin/freqtrade
OUT=results/sweep_v1_2
mkdir -p $OUT/logs
while read -r S; do
  [ -z "$S" ] && continue
  for M in binance_home kr_top10; do
    if [ "$M" = binance_home ]; then
      CFG=config_sweep_binance.json; DD=user_data/data/binance; TR=20251101-20261101; FEE=0.001
    else
      CFG=config_sweep_kr.json; DD=user_data/data_kr_rt/binance; TR=20260112-20261106; FEE=0.00115
    fi
    D=$OUT/$M/$S
    if [ -f "$D/.done" ]; then continue; fi
    mkdir -p $D
    echo "$(date -Is) start $M $S"
    if [ "$M" = binance_home ]; then RUN="$FT"; else RUN="$HOME/ftenv/bin/python ft_nofill.py"; fi
    nice -n 19 timeout 1800 $RUN backtesting -c $CFG --userdir user_data --datadir $DD --strategy "$S" ${TR:+--timerange $TR} --fee $FEE --cache none --export trades --backtest-directory $D > $OUT/logs/${M}__${S}.log 2>&1
    echo $? > $D/.exit
    touch $D/.done
    echo "$(date -Is) end $M $S exit $(cat $D/.exit)"
  done
done < sweep_strategies_v1.txt
echo "$(date -Is) sweep done"
