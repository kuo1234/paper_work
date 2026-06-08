# LIBERO isolated Docker setup

Purpose: install and test LIBERO without modifying the existing `bts_m1` CALVIN/3D-DA container.

## Why isolated

Spark currently has Docker and large disk, but no conda/micromamba/Python 3.8 environment manager visible. The existing `bts_m1` container uses Python 3.12 and should not be polluted with older LIBERO/robosuite dependencies.

## Build on spark

Expected repo location on spark:

```text
~/HDD/bts/paperwork
```

Run:

```bash
cd ~/HDD/bts/paperwork/docker/libero
bash build_and_run.sh
```

This builds:

```text
bts_libero:latest
```

and starts:

```text
container: bts_libero
```

## Smoke checks

After entering container:

```bash
docker exec -it bts_libero bash
python - <<'PY'
import libero
print('libero import ok', libero.__file__)
PY
```

Then inspect tasks/BDDL before running full render eval.

## Notes

- Dockerfile uses Python 3.10 rather than the older documented Python 3.8.13 because Ubuntu 22.04 includes Python 3.10. If LIBERO dependency conflicts appear, switch to a micromamba-based image with Python 3.8.13.
- `pip install -r requirements.txt || true` is intentionally permissive for first build. If build succeeds but runtime import fails, pin the failing packages in a follow-up commit.
- This setup is for LIBERO native diagnostics first. OpenVLA may require a separate newer stack or its own repository environment.
