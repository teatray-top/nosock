Set-Location G:\NoSock\stock\penny
function Run($envs, $log) {
  foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
  $p = Start-Process python -ArgumentList "-X", "utf8", "bt.py" -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  $p.PriorityClass = "Idle"; $p.WaitForExit()
  "== $log"; Get-Content $log | Select-String "fold|AGG"; Get-Content "$log.err" -Tail 3
}
foreach ($hd in 0, 2) {
  Run @{BT_IV = "all5m"; BT_H = "36"; BT_FOLD = "14"; BT_NFOLD = "3"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "$hd" } "runs\bt_all5m_head$hd.log"
  Run @{BT_IV = "all60m"; BT_H = "2"; BT_FOLD = "60"; BT_NFOLD = "6"; BT_VAL = "30"; BT_SEEDS = "20"; BT_HEAD = "$hd" } "runs\bt_all60m_head$hd.log"
  Run @{BT_IV = "us60m"; BT_H = "2"; BT_FOLD = "60"; BT_NFOLD = "6"; BT_VAL = "30"; BT_SEEDS = "20"; BT_HEAD = "$hd" } "runs\bt_us60m_head$hd.log"
}
