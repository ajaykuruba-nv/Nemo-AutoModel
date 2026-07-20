#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/userhome/home/akumarkuruba/rl/nemo-rl}"
CONFIG="${PROJECT_ROOT}/configs/grpo_gemma4_e4b_8xa100.yaml"

# Match prepare_data.sh: use an already-exported cluster secret first, otherwise
# load the project-local .env file and export all assignments from it.
if [[ -z "${HF_TOKEN:-}" && -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if [[ ! -s "${PROJECT_ROOT}/data/train.jsonl" || ! -s "${PROJECT_ROOT}/data/validation.jsonl" ]]; then
  echo "Prepared data is missing. Run scripts/prepare_data.sh first." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required to download the gated Gemma model." >&2
  echo "Add it to ${PROJECT_ROOT}/.env or export it in this shell." >&2
  exit 2
fi
if [[ "${HF_TOKEN}" == "hf_your_token_here" ]]; then
  echo "HF_TOKEN still contains the placeholder from .env.example." >&2
  exit 2
fi
if [[ ! -d "${NEMO_RL_ROOT}" ]]; then
  echo "NeMo RL not found at ${NEMO_RL_ROOT}. Set NEMO_RL_ROOT." >&2
  exit 2
fi

# The selected post-v0.6 NeMo RL checkout uses CUDA 13.2 PyTorch/vLLM wheels.
# On this A100 host the kernel driver advertises CUDA 12.8, so native execution
# needs NVIDIA's forward-compatibility user-mode driver libraries unless the
# host driver is upgraded. Ray workers inherit LD_LIBRARY_PATH from here.
CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/usr/local/cuda-13.2/compat}"
if [[ -e "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
  export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "Using CUDA compatibility libraries from ${CUDA_COMPAT_DIR}."
fi

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"

if ! uv run --project "${NEMO_RL_ROOT}" --extra nemo_gym \
  python -c 'import torch; assert torch.cuda.is_available(); print(f"CUDA ready: torch={torch.__version__}, devices={torch.cuda.device_count()}, gpu={torch.cuda.get_device_name(0)}")'; then
  echo "Native CUDA initialization failed." >&2
  echo "This checkout needs a CUDA-13-capable driver or cuda-compat-13-2." >&2
  echo "Ask the administrator to install cuda-compat-13-2, then verify ${CUDA_COMPAT_DIR}/libcuda.so.1 exists." >&2
  exit 2
fi

# Build/check the isolated dense worker environments before Ray starts. This
# avoids late, noisy policy-shutdown errors if an optional CUDA extension fails.
"${PROJECT_ROOT}/scripts/bootstrap_dense_envs.sh"

uv run --project "${NEMO_RL_ROOT}" --extra nemo_gym \
  python "${PROJECT_ROOT}/scripts/run_grpo_dense.py" \
  --config "${CONFIG}" "$@"
