#!/usr/bin/env bash
# Validate the native uv, Gemma 4, CUDA, and isolated worker environment.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/userhome/home/akumarkuruba/rl/nemo-rl}"
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/usr/local/cuda-13.2/compat}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or is not on PATH." >&2
  exit 2
fi
if [[ ! -f "${NEMO_RL_ROOT}/uv.lock" ]]; then
  echo "NeMo RL was not found at ${NEMO_RL_ROOT}." >&2
  exit 2
fi
if [[ -e "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
  export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "Using CUDA compatibility libraries from ${CUDA_COMPAT_DIR}."
else
  echo "CUDA compatibility library not found at ${CUDA_COMPAT_DIR}/libcuda.so.1."
  echo "Continuing in case the host driver was upgraded for CUDA 13."
fi

uv run --project "${NEMO_RL_ROOT}" --extra nemo_gym python -c \
  'from transformers import AutoConfig; AutoConfig.for_model("gemma4"); import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8, torch.cuda.device_count(); print(f"Main environment ready: torch={torch.__version__}, GPUs={torch.cuda.device_count()}, GPU 0={torch.cuda.get_device_name(0)}")'

"${PROJECT_ROOT}/scripts/bootstrap_dense_envs.sh"

VLLM_PY="${NEMO_RL_VENV_DIR:-${NEMO_RL_ROOT}/venvs}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
"${VLLM_PY}" -c \
  'from transformers import AutoConfig; AutoConfig.for_model("gemma4"); import torch, vllm; assert torch.cuda.is_available(); print(f"vLLM worker ready: vllm={vllm.__version__}, torch={torch.__version__}")'

echo "Native NeMo RL preflight passed."

