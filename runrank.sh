cd /g/NoSock/stock/penny
for a in "all5m 36" "all60m 2" "5m 36" "60m 2"; do
  READER_RANK=1 cmd //c "start /b /low /wait python reader.py $a 0.06 > runs_reader_tmp.out 2>&1"
  echo "== $a stop 0.06 rank"
  python -c "
import json
t=open('runs_reader_tmp.out').read(); print(t.splitlines()[0])
try:
    j=json.loads(t[t.index('{'):])
    for k in ('all','penny'): d=j[k]; print(k, 'n', d['n'], 'aucG %.3f'%d['auc_gain'], 'mean %+.2f%%'%(d['mean_ret']*100), 'top1 %+.2f%% w%.2f'%(d['top1%_ret']*100,d['top1%_win']), 'top5 %+.2f%% w%.2f'%(d['top5%_ret']*100,d['top5%_win']))
    for k,v in j.items():
        if k.startswith('best'): print(k, v)
except Exception as e: print(t[-1500:])
"
done
