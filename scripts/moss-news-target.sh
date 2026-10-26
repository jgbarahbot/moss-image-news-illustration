#!/bin/bash
# Target: MOSS-Image-SFT batch — N illustrations in ONE podman process
# (encoder 1x + DiT 1x + VAE 1x; N diffusion loops of 50 steps).
# Pattern: podman container with nvidia devices + driver libs mounted by
# hand (host has no nvidia-container-toolkit), batch mode (no server),
# no setsid (the wrapper's trap/timeout can kill it).
#
# Env (all have portable defaults; override per machine):
#   HERMES_DIR   default $HOME/.hermes
#   LOG          target log (default $HERMES_DIR/gpu-swaps/moss-news-target.log)
#   NAME         container name (default moss-news)
#   OUT          host output dir (default $HERMES_DIR/gpu-swaps/moss-out)
#   MODEL        host model dir (default $HERMES_DIR/models/moss-image-sft)
#   PROMPTS      host path to the prompts JSON — REQUIRED
#   INFER3       host path to infer3.py (default $HERMES_DIR/gpu-swaps/infer3.py;
#                copy scripts/infer3.py there on first use)
#   LIBS         dir with the nvidia driver libs (default /usr/lib/x86_64-linux-gnu)
#   PODMAN_IMAGE default moss-image:gpu
set -uo pipefail
HERMES_DIR="${HERMES_DIR:-$HOME/.hermes}"
GPU_DIR="$HERMES_DIR/gpu-swaps"
LOG="${LOG:-$GPU_DIR/moss-news-target.log}"
NAME="${NAME:-moss-news}"
OUT="${OUT:-$GPU_DIR/moss-out}"
MODEL="${MODEL:-$HERMES_DIR/models/moss-image-sft}"
PROMPTS="${PROMPTS:?set PROMPTS=/path/to/prompts.json}"
INFER3="${INFER3:-$GPU_DIR/infer3.py}"
LIBS="${LIBS:-/usr/lib/x86_64-linux-gnu}"
PODMAN_IMAGE="${PODMAN_IMAGE:-moss-image:gpu}"

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

mkdir -p "$OUT"
podman rm -f "$NAME" 2>/dev/null

# Conditional driver-lib mounts (some drivers lack ptxjitcompiler/cuvid/
# vdpau; hardcoding them gives rc=125).
DRIVER_LIBS=(libcuda.so.1 libnvidia-ml.so.1 libnvidia-cfg.so.1)
for f in libnvidia-ptxjitcompiler.so.1 libnvidia-gpu.so.1 libnvidia-ifr.so.1 \
         libnvcuvid.so.1 libnvvdpau.so.1; do
  [ -e "$LIBS/$f" ] && DRIVER_LIBS+=("$f")
done
VOLUMES=()
for f in "${DRIVER_LIBS[@]}"; do
  VOLUMES+=(-v "$LIBS/$f:/usr/lib/x86_64-linux-gnu/$f:ro")
done

log "subiendo contenedor $PODMAN_IMAGE (libs: ${DRIVER_LIBS[*]})"
podman run -d --name "$NAME" \
  --ipc host \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  "${VOLUMES[@]}" \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 \
  -e OMP_NUM_THREADS=4 \
  -e MOSS_H="1024" -e MOSS_W="1024" -e MOSS_STEPS="50" \
  -v "$MODEL":/model:ro \
  -v "$PROMPTS":/prompts.json:ro \
  -v "$OUT":/out \
  "$PODMAN_IMAGE" sleep 3600

# infer3.py from the host (loads encoder+DiT+VAE once, N generations).
podman cp "$INFER3" "$NAME":/opt/infer3.py
log "corriendo infer3 (batch, mismo grupo de procesos; sin setsid)"
podman exec "$NAME" python3 /opt/infer3.py 2>>"$LOG"
RC=$?
log "infer3 rc=$RC"

podman rm -f "$NAME" 2>/dev/null
exit $RC
