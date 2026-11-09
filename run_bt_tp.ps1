Set-Location G:\NoSock\stock\penny
while (Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -like '*run_bt_h.ps1*' }) { Start-Sleep -Seconds 30 }
$base = @{BT_FOLD = "14"; BT_NFOLD = "3"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "0"; BT_THRS = "0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"; DOBBY_FEE = "0.00015"; DOBBY_TAX = "0.0020"; DOBBY_US_FEE = "0.001"; DOBBY_US_SEC = "2.06e-5"; DOBBY_US_MIN_FEE = "0.01"; DOBBY_SLIP = "1.0" }
function Run($envs, $log) {
  foreach ($k in $base.Keys) { Set-Item -Path "env:$k" -Value $base[$k] }
  foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
  $p = Start-Process python -ArgumentList "-X", "utf8", "bt.py" -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  $p.PriorityClass = "Idle"; $p.WaitForExit()
  "== $log"; Get-Content $log | Select-String "fold|AGG|CHECK"; Get-Content "$log.err" -Tail 3
}
$gross = @{DOBBY_FEE = "0"; DOBBY_TAX = "0"; DOBBY_US_FEE = "0"; DOBBY_US_SEC = "0"; DOBBY_US_MIN_FEE = "0"; DOBBY_SLIP = "0"}
Run ($gross + @{BT_IV = "us5m"; BT_H = "6"; SCALP_TP = "0.008"; READER_STOP = "0.006"; BT_TAG = "bt_us5m_h6_tp0.008_gross"}) "runs\bt_us5m_h6_tp0.008_gross.log"
Run ($gross + @{BT_IV = "all5m"; BT_H = "6"; SCALP_TP = "0.01"; READER_STOP = "0.01"; BT_TAG = "bt_all5m_h6_tp0.01_gross"}) "runs\bt_all5m_h6_tp0.01_gross.log"
foreach ($h in 6, 12) {
  foreach ($tp in 0.008, 0.015) {
    Run @{BT_IV = "us5m"; BT_H = "$h"; SCALP_TP = "$tp"; READER_STOP = "0.006"; BT_TAG = "bt_us5m_h$h`_tp$tp" } "runs\bt_us5m_h$h`_tp$tp.log"
  }
}
foreach ($h in 6, 12) {
  foreach ($tp in 0.01, 0.02) {
    Run @{BT_IV = "all5m"; BT_H = "$h"; SCALP_TP = "$tp"; READER_STOP = "0.01"; BT_TAG = "bt_all5m_h$h`_tp$tp" } "runs\bt_all5m_h$h`_tp$tp.log"
  }
}
