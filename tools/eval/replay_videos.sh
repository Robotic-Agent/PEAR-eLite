#!/usr/bin/env bash
# usage: replay_videos.sh <run dir> [libero_replay_video.py args]: full 20 Hz videos of a run's recorded episodes
EVAL_DIR=$(cd "$(dirname "$0")" && pwd -P)
PEAR_ROOT=${PEAR_ROOT:-$(cd "$EVAL_DIR/../.." && pwd -P)}
WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
RUNS_DIR=${RUNS_DIR:-$WORK_DIR/runs}
M=$WORK_DIR/envs/mesa
RUN=$1
shift
exec env -i PATH=/usr/bin:/bin HOME=$HOME USER=${USER:-root} PYTHONPATH=$PEAR_ROOT \
  LIBERO_CONFIG_PATH=$WORK_DIR/envs/libero/etc/libero OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=$M/lib __EGL_VENDOR_LIBRARY_FILENAMES=$M/share/glvnd/egl_vendor.d/50_mesa.json \
  LIBGL_DRIVERS_PATH=$M/lib/dri GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4 \
  XDG_RUNTIME_DIR=/tmp/xdg_root CUDA_VISIBLE_DEVICES= \
  nice -n 10 $WORK_DIR/envs/libero/bin/python -u $EVAL_DIR/libero_replay_video.py "$RUN/episodes" "$@"
