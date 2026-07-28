#!/usr/bin/env bash
# Launch the retrieval server with the thread cap that keeps FAISS alive.
#
# WHY THE WRAPPER: serve_retrieval.py calls faiss.omp_set_num_threads(16), but by
# then `import torch` has already initialised OpenMP with this box's 192 threads,
# and FAISS then segfaults inside its SIMD kernels on the FIRST search — the
# server answers /health (index loaded) and dies the moment a query arrives, with
# an empty log, because a signal kill writes nothing. Verified 2026-07-28: 11
# threads segfaulting at one instruction pointer, and the identical search
# succeeds when OMP_NUM_THREADS is exported BEFORE the process starts.
# The in-code call is kept as a belt-and-braces second line of defence.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
# One held card is plenty: only the small E5 query encoder runs on GPU, the
# 64G flat index is searched on CPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[serve-retrieval] OMP_NUM_THREADS=$OMP_NUM_THREADS gpu=$CUDA_VISIBLE_DEVICES"
exec .venv-gpu3/bin/python scripts/serve_retrieval.py "$@"
