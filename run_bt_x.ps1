Set-Location G:\NoSock\stock\penny
while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like '*run_bt_h.ps1*' }) { Start-Sleep -Seconds 30 }
function Run($envs, $log) {
  foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
  $p = Start-Process python -ArgumentList "-X", "utf8", "bt.py" -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  $p.PriorityClass = "Idle"; $p.WaitForExit()
  "== $log"; Get-Content $log | Select-String "fold|AGG"; Get-Content "$log.err" -Tail 3
}
foreach ($h in 36, 12, 6) {
  foreach ($x in 0.1, 0.2) {
    Run @{BT_IV = "us5m"; BT_H = "$h"; BT_FOLD = "14"; BT_NFOLD = "3"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "0"; BT_EXIT = "$x" } "runs\bt_us5m_h$h`_x$x.log"
  }
}
Run @{BT_IV = "all5m"; BT_H = "36"; BT_FOLD = "14"; BT_NFOLD = "3"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "0"; BT_EXIT = "0.1" } "runs\bt_all5m_h36_x0.1.log"
