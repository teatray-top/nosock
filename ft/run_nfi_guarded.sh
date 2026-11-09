#!/usr/bin/env bash
set -u
cd /mnt/g/NoSock/stock/penny/ft
OUT=results/nfi_v1
mkdir -p $OUT
LIMIT_KB=$((16 * 1024 * 1024))
nice -n 19 ~/ftenv/bin/freqtrade backtesting -c nfi_configs/trading_mode-spot.json -c nfi_configs/exampleconfig.json -c nfi_configs/pairs_11.json -c nfi_configs/blacklist-binance.json --userdir user_data --datadir user_data/data/binance --strategy NostalgiaForInfinityX8 --timerange 20251101-20261101 --fee 0.001 --cache none --export trades --backtest-directory $OUT > $OUT/run.log 2>&1 &
PID=$!
PEAK=0
while kill -0 $PID 2>/dev/null; do
  RSS=0
  for p in $PID $(pgrep -P $PID); do
    r=$(awk '/VmRSS/ {print $2}' /proc/$p/status 2>/dev/null || echo 0)
    RSS=$((RSS + ${r:-0}))
  done
  [ $RSS -gt $PEAK ] && PEAK=$RSS
  if [ $RSS -gt $LIMIT_KB ]; then
    echo "$(date -Is) memory guard: RSS ${RSS} kB > ${LIMIT_KB} kB, killing" >> $OUT/guard.log
    kill $PID
    pkill -P $PID
    break
  fi
  sleep 5
done
wait $PID
EXIT=$?
echo "$(date -Is) exit $EXIT peak_rss_kb $PEAK" >> $OUT/guard.log
tail -1 $OUT/guard.log
