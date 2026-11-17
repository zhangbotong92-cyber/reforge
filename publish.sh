#!/usr/bin/env bash
# REFORGE - publish to GitHub (macOS / Linux twin of publish.ps1)
#
#   GITHUB_TOKEN=ghp_xxx ./publish.sh yourname [repo]
#   ./publish.sh yourname --dry-run
set -euo pipefail

OWNER="${1:?usage: GITHUB_TOKEN=ghp_xxx ./publish.sh <owner> [repo] [--public] [--dry-run]}"
REPO="${2:-reforge}"
BRANCH=main
API="https://api.github.com"
DRY=${3:-}

cd "$(dirname "$0")"

say() { printf '\033[36m[%s] %s\033[0m\n' "$1" "$2"; }
ok()  { printf '    %s\n' "$1"; }
die() { printf '\n  ERROR: %s\n' "$1" >&2; exit 1; }

say 1 "preflight"
command -v git >/dev/null || die "git not found"
curl -sfI "$API" >/dev/null || die "cannot reach $API"
[ -d .git ] || { git init -q; git checkout -q -b "$BRANCH"; }
git config user.email >/dev/null 2>&1 || git config user.email "reforge@users.noreply.github.com"
git config user.name  >/dev/null 2>&1 || git config user.name  "REFORGE"
ok "git $(git rev-parse --short HEAD 2>/dev/null || echo 'no commits')"

for f in README.md README.zh-CN.md pyproject.toml CONTRIBUTING.md .github/workflows/ci.yml; do
  [ -f "$f" ] && sed -i.bak "s|OWNER/$REPO|$OWNER/$REPO|g" "$f" && rm -f "$f.bak"
done
ok "owner placeholders stamped"

git add -A
git diff --cached --quiet || git commit -q -m "chore: stamp repository owner"
ok "$(git ls-files | wc -l | tr -d ' ') files, $(git rev-list --count HEAD 2>/dev/null || echo 1) commits"

if [ "$DRY" = "--dry-run" ]; then
  say 2 "dry run - nothing pushed"
  ok "would create $OWNER/$REPO and push $BRANCH"
  exit 0
fi

say 2 "authentication"
TOKEN="${GITHUB_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  read -r -s -p "    paste token (hidden): " TOKEN; echo
fi
[ -n "$TOKEN" ] || die "no token"
LOGIN=$(curl -sf -H "Authorization: Bearer $TOKEN" "$API/user" | sed -n 's/.*"login": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$LOGIN" ] || die "token rejected"
ok "authenticated as $LOGIN"

say 3 "create repository"
TARGET="users/$OWNER/repos"; [ "$LOGIN" = "$OWNER" ] || TARGET="orgs/$OWNER/repos"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/$TARGET" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"name\":\"$REPO\",\"description\":\"Reverse-engineer any AI image/video into a reproducible recipe\",\"private\":false}")
case "$CODE" in
  201) ok "created" ;;
  422) ok "already exists - pushing into it" ;;
  *)   die "create failed (HTTP $CODE)" ;;
esac

say 4 "push"
git remote remove origin 2>/dev/null || true
git remote add origin "https://x-access-token:$TOKEN@github.com/$OWNER/$REPO.git"
if git push -u origin "$BRANCH"; then
  git remote set-url origin "https://github.com/$OWNER/$REPO.git"
  ok "pushed, token scrubbed from .git/config"
else
  git remote set-url origin "https://github.com/$OWNER/$REPO.git"
  die "push failed (token already scrubbed)"
fi

printf '\n  LIVE -> https://github.com/%s/%s\n' "$OWNER" "$REPO"
