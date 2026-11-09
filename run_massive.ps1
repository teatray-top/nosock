Set-Location G:\NoSock\stock\penny
$base = @{BT_IV = "usm5m"; BT_FOLD = "14"; BT_NFOLD = "60"; BT_VAL = "7"; BT_SEEDS = "20"; BT_HEAD = "0"; BT_TRAIN_MIN = "120"; BT_TRAIN_MAX = "160"; BT_THRS = "0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9"; DOBBY_FEE = "0.00015"; DOBBY_TAX = "0.0020"; DOBBY_US_FEE = "0.001"; DOBBY_US_SEC = "2.06e-5"; DOBBY_US_MIN_FEE = "0.01"; DOBBY_SLIP = "1.0" }
$WIN = @{ w1 = @("2024-11-05", "2026-11-03"); w2 = @("2022-11-07", "2024-11-04"); w3 = @("2022-02-01", "2022-11-04") }
$cells = @{ A = @{BT_H = "12"; SCALP_TP = "0.015"; READER_STOP = "0.006"}; B = @{BT_H = "36"; SCALP_TP = "0"; READER_STOP = "0.06"}; C = @{BT_H = "12"; SCALP_TP = "0.008"; READER_STOP = "0.006"}; D = @{BT_H = "6"; SCALP_TP = "0.015"; READER_STOP = "0.006"} }
function Idle($args_, $log) {
  $p = Start-Process python -ArgumentList (@("-X", "utf8") + $args_) -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
  $p.PriorityClass = "Idle"; $p.WaitForExit(); $p.ExitCode
}
function Panel($w) {
  $out = "data\panelusm5m_$w`_v4.npz"
  if (Test-Path $out) { return $out }
  while (-not (Test-Path "data\massive\flat\minute\$($WIN[$w][0]).csv.gz")) { Start-Sleep -Seconds 60 }
  Start-Sleep -Seconds 30
  Write-Host "== panel $w"
  $null = Idle @("panel_massive.py", $WIN[$w][0], $WIN[$w][1], $out, "5m", "1000") "runs\panel_usm5m_$w.log"
  Get-Content "runs\panel_usm5m_$w.log" -Tail 2 | Write-Host
  Get-Content "runs\panel_usm5m_$w.log.err" -Tail 3 | Write-Host
  return $out
}
function Run($w, $c) {
  $out = Panel $w
  if (-not (Test-Path $out)) { "panel $w missing, skip"; return }
  foreach ($k in $base.Keys) { Set-Item -Path "env:$k" -Value $base[$k] }
  foreach ($k in $cells[$c].Keys) { Set-Item -Path "env:$k" -Value $cells[$c][$k] }
  $env:PANEL_PATH = (Resolve-Path $out).Path
  $env:BT_TAG = "bt_usm5m_$w`_$c"
  $log = "runs\bt_usm5m_$w`_$c.log"
  if (Test-Path $log) { if (Select-String -Path $log -Pattern "^AGG" -Quiet) { "== $log already done"; return } }
  "== $log start $(Get-Date -Format HH:mm) panel $env:PANEL_PATH"
  $null = Idle @("bt.py") $log
  Get-Content $log | Select-String "AGG|CHECK|too short"; Get-Content "$log.err" -Tail 3
}
foreach ($w in "w1", "w2", "w3") { Run $w "A" }
foreach ($w in "w1", "w2", "w3") { Run $w "B" }
Run "w1" "C"
Run "w1" "D"
"all done $(Get-Date -Format HH:mm)"
