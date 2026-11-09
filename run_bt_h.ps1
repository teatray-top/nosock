Set-Location G:\NoSock\stock\penny
function Run($envs, $log) {
  foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
  $p = Start-Process python -ArgumentList "-X", "utf8", "bt.py" -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  $p.PriorityClass = "Idle"; $p.WaitForExit()
  "== $log"; Get-Content $log | Select-String "fold|AGG"; Get-Content "$log.err" -Tail 3
}
foreach ($h in 6, 12, 36) {
  Run @{BT_IV = "us5m"; BT_H = "$h"; BT_FOLD = "14"; BT_NFOLD = "3"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "0" } "runs\bt_us5m_h$h`_floor.log"
}
foreach ($h in 6, 12) {
  Run @{BT_IV = "all5m"; BT_H = "$h"; BT_FOLD = "14"; BT_NFOLD = "3"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "0" } "runs\bt_all5m_h$h`_floor.log"
}
