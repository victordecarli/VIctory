<#
.SYNOPSIS
Builds victory and runs the configured title for a test, with only the
switches given here, then summarises the run.

.DESCRIPTION
1. Builds the diag build (release code plus the diagnostics feature, in
   target\diag); -Release builds and runs the plain release build instead.
   An unchanged tree builds in seconds, so a test never runs an old binary.
2. Runs `victory run --execute` from the repository root, so victory.toml
   is found; the title is the only .exe at the top of the configured
   game_dir unless -Exe names another.
3. The VICTORY_* switches (and RUST_LOG) the session already has are left
   out of the run and listed, unless -KeepEnv; the configuration variables
   (VICTORY_CONFIG, VICTORY_GAME_DIR, VICTORY_CACHE_DIR, VICTORY_DATA_DIR,
   VICTORY_LOG_DIR, VICTORY_TEST_TITLES) always pass. The session's environment is the same
   afterwards.
4. Waits for every victory.exe the run started (a relaunched title, the
   audio recorder), then shows scripts\run-summary.ps1 for the run.

The switches are the README's (Debugging). A release build has none of
the debugging aids, so most of them do nothing with -Release.

.EXAMPLE
scripts\run.ps1
.EXAMPLE
scripts\run.ps1 -Warp
Renders on the software adapter (the README's advice for entering the world).
.EXAMPLE
scripts\run.ps1 -GpuDebug -Log 'victory_gpu=debug,info'
.EXAMPLE
scripts\run.ps1 -Keys '30:Enter,35:X' -Env @{ VICTORY_STACKS = '20,60' }
.EXAMPLE
scripts\run.ps1 -Release -NoBuild
#>
[CmdletBinding()]
param(
    # Build and run the plain release build (the README's) instead of diag.
    [switch]$Release,
    # Run the build that is there, without building.
    [switch]$NoBuild,
    # VICTORY_D3D12_WARP=1: the host device on the software adapter.
    [switch]$Warp,
    # VICTORY_D3D12_DEBUG=1 and VICTORY_D3D12_DRED=1: the host debug layer,
    # and the allocation and commands of a GPU failure.
    [switch]$GpuDebug,
    # VICTORY_D3D12_DEBUG=1 and VICTORY_D3D12_GBV=1: GPU-based validation (slow).
    [switch]$Gbv,
    # VICTORY_LOG: the log filter, e.g. 'victory_gpu=debug,info'.
    [string]$Log,
    # VICTORY_KEYS: keys pressed at those seconds, e.g. '30:Enter,35:X'.
    [string]$Keys,
    # Any other switch for this run, e.g. @{ VICTORY_STACKS = '20,60' }.
    [hashtable]$Env = @{},
    # Keep the VICTORY_* switches the session already has.
    [switch]$KeepEnv,
    # The console model the title sees.
    [ValidateSet('series-x', 'series-s')]
    [string]$Console = 'series-x',
    # Another executable than the game_dir's.
    [string]$Exe,
    # Arguments for the title.
    [string[]]$TitleArgs = @()
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$build = if ($Release) { 'release' } else { 'diag' }
$victory = Join-Path $repo "target\$build\victory.exe"

Push-Location $repo
try {
    if (-not $NoBuild) {
        if ($Release) {
            cargo build --release -p victory --locked
        } else {
            cargo build --profile diag -p victory --features diagnostics --locked
        }
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
    }
    if (-not (Test-Path $victory)) { throw "$victory does not exist; run without -NoBuild" }

    # The switches of this run.
    $wanted = [ordered]@{}
    if ($Warp) { $wanted.VICTORY_D3D12_WARP = '1' }
    if ($GpuDebug) { $wanted.VICTORY_D3D12_DEBUG = '1'; $wanted.VICTORY_D3D12_DRED = '1' }
    if ($Gbv) { $wanted.VICTORY_D3D12_DEBUG = '1'; $wanted.VICTORY_D3D12_GBV = '1' }
    if ($Log) { $wanted.VICTORY_LOG = $Log }
    if ($Keys) { $wanted.VICTORY_KEYS = $Keys }
    foreach ($name in $Env.Keys) { $wanted[$name] = [string]$Env[$name] }

    # Inherited switches are left out of the run; the session gets them back.
    $config = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_LOG_DIR', 'VICTORY_TEST_TITLES'
    $saved = @{}
    if (-not $KeepEnv) {
        $inherited = Get-ChildItem Env: |
            Where-Object { ($_.Name -like 'VICTORY_*' -or $_.Name -eq 'RUST_LOG') -and $_.Name -notin $config }
        foreach ($item in $inherited) {
            $saved[$item.Name] = $item.Value
            Remove-Item "Env:$($item.Name)"
        }
        if ($inherited) {
            Write-Host "Left out of this run (session switches; -KeepEnv keeps them): $(($inherited | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ', ')" -ForegroundColor Yellow
        }
    }
    foreach ($name in $wanted.Keys) {
        if (-not $saved.ContainsKey($name)) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
        Set-Item "Env:$name" $wanted[$name]
    }

    $shown = if ($wanted.Count) { ($wanted.Keys | ForEach-Object { "$_=$($wanted[$_])" }) -join ', ' } else { 'none' }
    Write-Host "Running $build build, switches: $shown" -ForegroundColor Cyan
    if ($Release -and ($wanted.Keys | Where-Object { $_ -ne 'VICTORY_LOG' })) {
        Write-Host 'A release build has no debugging aids (README, Debugging): these switches may do nothing.' -ForegroundColor Yellow
    }

    $arguments = @('run')
    if ($Exe) { $arguments += $Exe }
    $arguments += '--execute', '--console', $Console
    if ($TitleArgs) { $arguments += '--'; $arguments += $TitleArgs }

    $started = Get-Date
    $exit = 0
    try {
        & $victory @arguments
        $exit = $LASTEXITCODE
        # A relaunched title and the audio recorder are victory processes of
        # their own; the run ends with the last of them.
        while ($true) {
            $later = @(Get-Process -Name victory -ErrorAction SilentlyContinue |
                Where-Object { $_.Path -ieq $victory -and $_.StartTime -ge $started })
            if (-not $later) { break }
            Write-Host "Waiting for victory $(($later | ForEach-Object Id) -join ', ') (relaunched title or audio recorder)" -ForegroundColor Cyan
            $later | Wait-Process
        }
    } finally {
        foreach ($name in $saved.Keys) {
            if ($null -eq $saved[$name]) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
            else { Set-Item "Env:$name" $saved[$name] }
        }
    }

    Write-Host ''
    Write-Host "victory exited with $exit" -ForegroundColor $(if ($exit -eq 0) { 'Green' } else { 'Red' })
    & (Join-Path $PSScriptRoot 'run-summary.ps1') -Since $started -Victory $victory
    exit $exit
} finally {
    Pop-Location
}
