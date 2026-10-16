# auto_sync.ps1 - Real-time bidirectional sync between VS Code and GitHub
param(
    [int]$IntervalSeconds = 10
)

$repoPath = $PSScriptRoot
Set-Location $repoPath

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   REAL-TIME AUTO-SYNC: VS Code <---> GitHub      " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Watching: $repoPath" -ForegroundColor White
Write-Host "Sync interval: Every $IntervalSeconds seconds" -ForegroundColor Gray
Write-Host "Status: Active (Press Ctrl+C to pause/stop)" -ForegroundColor Yellow
Write-Host "--------------------------------------------------" -ForegroundColor Cyan

while ($true) {
    try {
        # Step 1: Check for remote changes on GitHub (GitHub -> VS Code)
        git fetch origin main 2>$null
        $status = git status -uno 2>$null
        if ($status -match "Your branch is behind") {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Incoming changes from GitHub! Pulling to VS Code..." -ForegroundColor Cyan
            git pull --rebase origin main
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Local VS Code files updated from GitHub!" -ForegroundColor Green
        }

        # Step 2: Check for local changes made in VS Code (VS Code -> GitHub)
        $diff = git status --porcelain 2>$null
        if ($diff) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Local edits detected in VS Code! Syncing to GitHub..." -ForegroundColor Yellow
            git add .
            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            git commit -m "Auto-sync update: $timestamp"
            git push origin main
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Pushed to GitHub successfully!" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "Sync check notice: $_" -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds $IntervalSeconds
}
