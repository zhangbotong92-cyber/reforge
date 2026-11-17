# REFORGE — publish to GitHub in one command.
#
#   .\publish.ps1 -Owner yourname                      # will prompt for the token
#   $env:GITHUB_TOKEN="ghp_xxx"; .\publish.ps1 -Owner yourname
#   .\publish.ps1 -Owner yourname -DryRun               # preflight only, no auth needed
#
# Token needs the `repo` scope (classic) or Contents+Administration write (fine-grained).
# Safe to re-run: reuses an existing repo, rewrites OWNER placeholders, never force-pushes
# unless you pass -Force. The token is scrubbed from .git/config afterwards.

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Owner,
  [string]$Repo = "reforge",
  [string]$Token = $env:GITHUB_TOKEN,
  [string]$Branch = "main",
  [switch]$Public,
  [switch]$Force,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$api = "https://api.github.com"

function Step($n, $msg) { Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "    $msg" -ForegroundColor DarkGray }
function Die($msg) { Write-Host "`n  ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- preflight
Step 1 "preflight"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die "git not found" }
try {
  $null = Invoke-WebRequest -Uri "$api" -UseBasicParsing -TimeoutSec 10 -Method Head
  Ok "network: api.github.com reachable"
} catch { Die "cannot reach api.github.com: $($_.Exception.Message)" }

if (-not (Test-Path ".git")) {
  git init -q
  git checkout -q -b $Branch
  Ok "git repo initialised"
}
if (-not (git config user.email)) { git config user.email "reforge@users.noreply.github.com" }
if (-not (git config user.name)) { git config user.name "REFORGE" }
Ok "git $(git rev-parse --short HEAD 2>$null) on $(git rev-parse --abbrev-ref HEAD)"

# stamp the real owner into every OWNER placeholder
$stamped = 0
foreach ($f in @("README.md", "README.zh-CN.md", "pyproject.toml", "CONTRIBUTING.md", ".github/workflows/ci.yml")) {
  if (Test-Path $f) {
    $raw = Get-Content $f -Raw
    if ($raw -match "OWNER/$Repo") {
      ($raw -replace "OWNER/$Repo", "$Owner/$Repo") | Set-Content $f -NoNewline -Encoding utf8
      $stamped++
    }
  }
}
Ok "owner placeholders stamped in $stamped files"

git add -A
if (git status --porcelain) {
  git commit -q -m "chore: stamp repository owner" 2>$null
  Ok "committed pending changes"
}
$tracked = (git ls-files).Count
Ok "$tracked files tracked, $((git rev-list --count HEAD)) commits"

if ($DryRun) {
  Step 2 "dry run (no authentication, nothing is pushed)"
  Ok "would create : $Owner/$Repo (public=$([bool]$Public))"
  Ok "would push   : HEAD -> origin/$Branch"
  Ok "would set    : description, homepage, topics"
  Ok "token source : $(if ($Token) {'provided'} else {'would prompt'})"
  Write-Host "`npreflight OK - rerun without -DryRun to publish." -ForegroundColor Green
  exit 0
}

# ---------------------------------------------------------------- token
Step 2 "authentication"
if (-not $Token) {
  Write-Host "    no GITHUB_TOKEN in the environment." -ForegroundColor Yellow
  Write-Host "    create one: https://github.com/settings/tokens/new  (scope: repo)" -ForegroundColor Yellow
  try {
    $sec = Read-Host "    paste token (input hidden, Enter to abort)" -AsSecureString
    $Token = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
      [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
  } catch { $Token = Read-Host "    paste token" }
}
if (-not $Token) { Die "no token supplied" }

$headers = @{
  Authorization          = "Bearer $Token"
  Accept                 = "application/vnd.github+json"
  "X-GitHub-Api-Version" = "2022-11-28"
  "User-Agent"           = "reforge-publish"
}
try {
  $resp = Invoke-WebRequest -Uri "$api/user" -Headers $headers -UseBasicParsing -TimeoutSec 20
  $user = $resp.Content | ConvertFrom-Json
} catch { Die "token rejected: $($_.Exception.Message)" }
Ok "authenticated as $($user.login)"

$scopes = $resp.Headers["x-oauth-scopes"]
if ($scopes -and $scopes -notmatch "repo") {
  Write-Host "    warning: token scopes are '$scopes' - 'repo' is required to create and push" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- repo
Step 3 "create repository"
$target = if ($user.login -eq $Owner) { "users/$Owner/repos" } else { "orgs/$Owner/repos" }
$body = @{
  name        = $Repo
  description = "Reverse-engineer any AI image/video into a reproducible recipe: prompt + controls + pins, stamped back into the file."
  homepage    = "https://github.com/$Owner/$Repo"
  private     = (-not $Public)
  has_issues  = $true
  has_wiki    = $false
} | ConvertTo-Json

try {
  $created = Invoke-RestMethod -Uri "$api/$target" -Method Post -Headers $headers -Body $body -ContentType "application/json"
  Ok "created $($created.full_name)"
} catch {
  $code = $_.Exception.Response.StatusCode.value__
  if ($code -eq 422) { Ok "already exists - pushing into it" }
  elseif ($code -eq 404 -and $user.login -ne $Owner) { Die "no access to org '$Owner'" }
  else { Die "create failed ($code): $($_.Exception.Message)" }
}

# ---------------------------------------------------------------- push
Step 4 "push"
$authUrl = "https://x-access-token:$Token@github.com/$Owner/$Repo.git"
$cleanUrl = "https://github.com/$Owner/$Repo.git"
if (git remote get-url origin 2>$null) { git remote set-url origin $authUrl } else { git remote add origin $authUrl }

$pushArgs = @("push", "-u", "origin", $Branch)
if ($Force) { $pushArgs += "--force" }
& git @pushArgs
$code = $LASTEXITCODE

# always scrub the token out of .git/config, success or failure
git remote set-url origin $cleanUrl
if ($code -ne 0) { Die "push failed (exit $code); token already removed from .git/config" }
Ok "pushed HEAD -> origin/$Branch (token scrubbed from .git/config)"

# ---------------------------------------------------------------- polish
Step 5 "metadata"
$topics = @{
  names = @("reverse-prompt", "image-to-prompt", "reproducibility", "stable-diffusion", "comfyui",
            "forensics", "c2pa", "prompt-engineering", "video-generation", "python")
} | ConvertTo-Json
try {
  Invoke-RestMethod -Uri "$api/repos/$Owner/$Repo/topics" -Method Put -Headers $headers `
    -Body $topics -ContentType "application/json" | Out-Null
  Ok "10 topics set"
} catch { Ok "topics skipped: $($_.Exception.Message)" }

Write-Host ""
Write-Host "  LIVE -> https://github.com/$Owner/$Repo" -ForegroundColor Green
Write-Host "  next: pin the repo, then post the L1/L2/L3 table as the launch note." -ForegroundColor DarkGray
