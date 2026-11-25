<#
.SYNOPSIS
Summarises a VIctory run from its run logs.

.DESCRIPTION
Reads the run logs (<log_dir>\run-*.jsonl, log_dir as `victory config`
prints it) and shows, for each process of the run: its role and
switches, the host adapter, the last frame report, the costliest dropped
draws, a crash, and the last summary line. Only lines with the target
victory::summary are read.

scripts\run.ps1 calls it when the title ends. Run it alone to look at the
newest run again.

.PARAMETER Since
The logs written since this time: every process of one run (a relaunch and
the audio recorder write their own). Default: the newest log only.

.PARAMETER Victory
The victory.exe to ask for the log folder. Default: target\diag, else
target\release.

.PARAMETER Top
How many of the costliest drops to list.

.EXAMPLE
scripts\run-summary.ps1
#>
[CmdletBinding()]
param(
    [datetime]$Since,
    [string]$Victory,
    [int]$Top = 10
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

if (-not $Victory) {
    $Victory = @('diag', 'release') |
        ForEach-Object { Join-Path $repo "target\$_\victory.exe" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    if (-not $Victory) { throw 'No victory.exe in target\diag or target\release: build it first (scripts\run.ps1 does).' }
}

# The log folder as the program resolves it (victory.toml is read from the
# repository root, as scripts\run.ps1 runs it).
Push-Location $repo
try { $config = & $Victory config } finally { Pop-Location }
$logLine = $config | Where-Object { $_ -match '^log_dir\s*:\s*(.+)$' } | Select-Object -First 1
if (-not $logLine) { throw "``victory config`` printed no log_dir:`n$($config -join "`n")" }
$logs = ($logLine -replace '^log_dir\s*:\s*', '').Trim()

$files = @(Get-ChildItem -Path $logs -Filter 'run-*.jsonl' -ErrorAction SilentlyContinue | Sort-Object Name)
if ($Since) {
    $files = @($files | Where-Object { $_.LastWriteTime -ge $Since })
} elseif ($files) {
    $files = @($files[-1])
}
if (-not $files) {
    Write-Host "No run log in $logs$(if ($Since) { " since $Since" })."
    return
}

function Read-Summary([System.IO.FileInfo]$File) {
    Select-String -Path $File.FullName -SimpleMatch '"victory::summary"' | ForEach-Object {
        try { $_.Line | ConvertFrom-Json }
        catch { Write-Warning "$($File.Name):$($_.LineNumber): not JSON, skipped" }
    }
}

function Show-Drops($Events) {
    $start = -1
    for ($i = $Events.Count - 1; $i -ge 0; $i--) {
        if ($Events[$i].message -eq 'drop totals') { $start = $i; break }
    }
    $firsts = @($Events | Where-Object { $_.message -like 'first drop*' }).Count
    if ($start -lt 0) {
        Write-Host "  drops: none reported$(if ($firsts) { " ($firsts first drops)" })"
        return
    }
    $totals = $Events[$start]
    Write-Host "  drops (run totals at frame $($totals.frame); $($totals.shaders) shaders, $($totals.pipelines) pipelines):"
    $items = for ($i = $start + 1; $i -lt $Events.Count -and $Events[$i].message -like 'drop total*'; $i++) {
        $e = $Events[$i]
        switch ($e.message) {
            'drop total: shader' {
                [pscustomobject]@{ Drops = [long]$e.drops; What = "shader $($e.stage) $($e.hash) ($($e.class))" }
            }
            'drop total: pipeline' {
                $sum = [long]$e.being_made + [long]$e.refused + [long]$e.unprepared
                [pscustomobject]@{ Drops = $sum; What = "pipeline $($e.pipeline) [$($e.shaders)] being made $($e.being_made), refused $($e.refused), unprepared $($e.unprepared)" }
            }
            default {
                [pscustomobject]@{ Drops = [long]$e.drops; What = $e.cause }
            }
        }
    }
    $items | Sort-Object Drops -Descending | Select-Object -First $Top | ForEach-Object {
        Write-Host ('    {0,10}  {1}' -f $_.Drops, $_.What)
    }
}

foreach ($file in $files) {
    $events = @(Read-Summary $file)
    $start = $events | Where-Object { $_.message -eq 'run_start' } | Select-Object -First 1
    Write-Host ''
    Write-Host "== $($file.FullName)"
    if ($start) {
        Write-Host "  $($start.role), pid $($start.pid), diagnostics $($start.diagnostics), switches $($start.switches)"
    }
    $adapter = $events | Where-Object { $_.message -eq 'host device adapter' } | Select-Object -First 1
    if ($adapter) { Write-Host "  adapter: $($adapter.adapter)" }
    $frame = $events | Where-Object { $_.message -match '^\S+ fps, ' } | Select-Object -Last 1
    if ($frame) { Write-Host "  last frame report: $($frame.message)" }
    Show-Drops $events
    foreach ($crash in @($events | Where-Object { $_.message -eq 'crash' })) {
        Write-Host '  CRASH:' -ForegroundColor Red
        ($crash.report -split "`n" | Select-Object -First 15) | ForEach-Object { Write-Host "    $_" }
    }
    $last = $events | Select-Object -Last 1
    if ($last) { Write-Host "  last line: $($last.timestamp) $($last.message)" }
}
