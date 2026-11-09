param([string]$mk = "kr", [switch]$dry)
Set-Location G:\NoSock\stock\penny
$day = Get-Date -Format yyyyMMdd
$env:TOSS_RPS = "8"
if ($mk -eq "kr") { $env:WS_EXTRA = "069500,114800" }
$procs = @()
$p = Start-Process python -ArgumentList "-X", "utf8", "record_ws.py", $mk -RedirectStandardOutput "data\toss\ws\record_$mk.out" -RedirectStandardError "data\toss\ws\record_$mk.err" -PassThru -NoNewWindow
$p.PriorityClass = "BelowNormal"
$procs += $p
"$(Get-Date -Format 'yyyy-MM-dd HH:mm') $mk recorder pid $($p.Id)" | Add-Content runs\record_sched.log
if ($dry) {
  Remove-Item Env:DOBBY_LIVE -ErrorAction SilentlyContinue
  $q = Start-Process python -ArgumentList "-X", "utf8", "live.py", $mk -RedirectStandardOutput "runs\live_dry_$mk`_$day.out" -RedirectStandardError "runs\live_dry_$mk`_$day.err" -PassThru -NoNewWindow
  $q.PriorityClass = "BelowNormal"
  $procs += $q
  "$(Get-Date -Format 'yyyy-MM-dd HH:mm') $mk dry loop pid $($q.Id)" | Add-Content runs\record_sched.log
}
$procs | ForEach-Object { $_.WaitForExit() }
"$(Get-Date -Format 'yyyy-MM-dd HH:mm') $mk done" | Add-Content runs\record_sched.log
