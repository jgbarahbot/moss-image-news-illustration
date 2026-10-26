#!/bin/bash
# GPU-swap wrapper: OFF (the serving LLM) -> TARGET -> ON (always, via trap).
#
# The agent runs on the LLM being swapped out, so it cannot answer while this
# is in flight — the caller is the human's message turn; launch and move on.
#
# Env:
#   HERMES_DIR    default $HOME/.hermes
#   OFF / ON      scripts that stop / restart the serving LLM
#                 (default $HOME/hyperqwen/hyperqwen_{off,on}.sh)
#   LOG           default $HERMES_DIR/gpu-swaps/moss-news-swap.log
#   TARGET_CMD    command of the target phase; defaults to the bundled
#                 moss-news-target.sh with PROMPTS forwarded
#   TARGET_TIMEOUT_S  default 1800 (safety net; target normally ~16 min/10 imgs)
set -uo pipefail
HERMES_DIR="${HERMES_DIR:-$HOME/.hermes}"
GPU_DIR="$HERMES_DIR/gpu-swaps"
LOG="${LOG:-$GPU_DIR/moss-news-swap.log}"
OFF="${OFF:-$HOME/hyperqwen/hyperqwen_off.sh}"
ON="${ON:-$HOME/hyperqwen/hyperqwen_on.sh}"
LOCK="${LOCK:-/tmp/gpu-swap.lock}"
TARGET_CMD="${TARGET_CMD:-env PROMPTS=$PROMPTS bash $GPU_DIR/moss-news-target.sh}"
# Estimate: encoder 1x ~2.5 min + N x (50 steps ~1.5 min + VAE ~1 min) +
# weight load ~2 min; timeout is a safety net, not an expectation.
TARGET_TIMEOUT_S="${TARGET_TIMEOUT_S:-1800}"

mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# --- lock: no two swaps at once (if the lock is not mine, DO NOT touch it) ---
if [ -f "$LOCK" ] && [ -n "$(cat "$LOCK" 2>/dev/null)" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  log "FATAL: ya hay otro GPU swap en marcha (pid $(cat "$LOCK")); aborto sin tocar nada."
  exit 1
fi
echo $$ > "$LOCK"

on_ran=0
cleanup(){
  if [ -n "${T_PID:-}" ] && kill -0 "$T_PID" 2>/dev/null; then
    log "cleanup: target todavia vivo (pid $T_PID); matandolo antes del ON"
    kill -TERM "$T_PID" 2>/dev/null || true
    pkill -TERM -f "$TARGET_CMD" 2>/dev/null || true
    sleep 5
    kill -KILL "$T_PID" 2>/dev/null || true
    pkill -KILL -f "$TARGET_CMD" 2>/dev/null || true
  fi
  if [ "$on_ran" != 1 ]; then
    log "===== FASE ON (via trap: el target fallo o fue matado) ====="
    bash "$ON" >>"$LOG" 2>&1
    log "fase ON terminada (rc=$?)"
    on_ran=1
  fi
  rm -f "$LOCK"
}
trap cleanup EXIT INT TERM

log "===== FASE OFF ($(date '+%F %T')) ====="
bash "$OFF" >>"$LOG" 2>&1
rc=$?
if [ "$rc" != 0 ]; then
  log "FATAL: el OFF fallo (rc=$rc); el modelo sigue arriba; no ejecuto el target."
  exit 1
fi

log "===== FASE TARGET: $TARGET_CMD (max ${TARGET_TIMEOUT_S}s) ====="
if [ "$TARGET_TIMEOUT_S" -gt 0 ] 2>/dev/null; then
  timeout --signal=TERM --kill-after=15s "${TARGET_TIMEOUT_S}s" bash -c "$TARGET_CMD" >>"$LOG" 2>&1 &
else
  bash -c "$TARGET_CMD" >>"$LOG" 2>&1 &
fi
T_PID=$!
wait "$T_PID"
rc=$?
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
  log "target EXCEDIO los ${TARGET_TIMEOUT_S}s (rc=$rc): cortado (TERM, y KILL si resistio); VRAM liberada; paso al ON de todos modos."
else
  log "target terminado (rc=$rc)"
fi

log "===== FASE ON ($(date '+%F %T')) ====="
bash "$ON" >>"$LOG" 2>&1
rc_on=$?
on_ran=1
log "fase ON terminada (rc=$rc_on)"
log "===== SWAP COMPLETO: target rc=$rc, on rc=$rc_on ====="
exit $rc
