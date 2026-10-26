#!/usr/bin/env bash
# Sync a local Hermes skill to its GitHub repo (<owner>/<skill>).
# Commits local changes, fast-forwards, pushes.
#
# Parameterized GitHub identity — nothing hardcoded. Resolution order
# (CLI env > env-file > derived from the token):
#   SKILL_TOKEN_ENV   name of the env var holding the push token
#                     (default GITHUB_CONTENT_RW_TOKEN)
#   SKILL_REPO        owner/name of the remote
#                     (default <token-login>/<skill-dir-name>)
#   SKILL_GH_USER     git commit author username
#                     (default: token's own login, via the /user API)
#
# Usage: ./sync.sh [commit message]
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL="$(basename "$DIR")"
cd "$DIR"

TOKEN_ENV="${SKILL_TOKEN_ENV:-GITHUB_CONTENT_RW_TOKEN}"
TOK="$(grep "^${TOKEN_ENV}=" "$HOME/.hermes/.env" 2>/dev/null | head -n1 | cut -d= -f2- | tr -d '\r')"
[ -n "$TOK" ] || { echo "ERROR: ${TOKEN_ENV} not found in ~/.hermes/.env" >&2; exit 1; }

# Derive the owning GitHub login from the token (single source of truth).
GH_USER="$(curl -sf -H "Authorization: Bearer $TOK" -H "Accept: application/vnd.github+json" \
    https://api.github.com/user 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('login','') or '')")"
[ -n "$GH_USER" ] || { echo "ERROR: could not resolve the token's GitHub login" >&2; exit 1; }
GH_USER="${SKILL_GH_USER:-$GH_USER}"

REPO="${SKILL_REPO:-$GH_USER/$SKILL}"
URL="https://${GH_USER}:${TOK}@github.com/${REPO}.git"

git add -A
if ! git diff --cached --quiet; then
  git -c user.name="$GH_USER" -c user.email="${GH_USER}@users.noreply.github.com" \
    commit -m "${1:-sync: $SKILL $(date -u +%F)}"
  echo "committed local changes"
fi
git fetch "$URL" main
if ! git merge --ff-only FETCH_HEAD >/dev/null 2>&1; then
  echo "ERROR: cannot fast-forward; resolve diverged history manually" >&2
  exit 1
fi
git push "$URL" main
echo "synced $SKILL -> $REPO (up to date)"
