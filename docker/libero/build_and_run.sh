#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$HOME/HDD/bts}
IMAGE=${IMAGE:-bts_libero:latest}
CONTAINER=${CONTAINER:-bts_libero}

cd "$ROOT/paperwork/docker/libero"
docker build -t "$IMAGE" .

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker rm -f "$CONTAINER" >/dev/null
fi

docker run -dit \
  --gpus all \
  --name "$CONTAINER" \
  --ipc=host \
  --shm-size=16g \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -v "$ROOT:/workspace/bts" \
  "$IMAGE"

echo "started $CONTAINER from $IMAGE"
echo "enter with: docker exec -it $CONTAINER bash"
