#!/usr/bin/env bash
# Build the two isolated NeMo RL worker environments needed by dense Gemma 4.
#
# NeMo RL's broad `automodel` and `vllm` extras include optional MoE, Mamba,
# DeepEP, DeepGEMM, and Transformer Engine source builds. Those require a full
# CUDA toolkit/nvcc even though this dense, PyTorch-backend recipe never imports
# them. Keep the locked dependency graph but omit only those unused packages.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/userhome/home/akumarkuruba/rl/nemo-rl}"
VENV_ROOT="${NEMO_RL_VENV_DIR:-${NEMO_RL_ROOT}/venvs}"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap is validated for the requested x86_64 A100 host." >&2
  exit 2
fi
if [[ ! -f "${NEMO_RL_ROOT}/uv.lock" ]]; then
  echo "NeMo RL with uv.lock was not found at ${NEMO_RL_ROOT}." >&2
  exit 2
fi

VLLM_VENV="${VENV_ROOT}/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
AUTOMODEL_VENV="${VENV_ROOT}/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

mkdir -p "${VENV_ROOT}"

if [[ -x "${VLLM_VENV}/bin/python" ]] && \
  "${VLLM_VENV}/bin/python" -c \
    'from transformers import AutoConfig; AutoConfig.for_model("gemma4"); import torch, vllm' \
    >/dev/null 2>&1; then
  echo "Reusing ready dense vLLM worker environment."
else
  echo "Preparing dense vLLM worker environment..."
  uv venv --allow-existing "${VLLM_VENV}"
  UV_PROJECT_ENVIRONMENT="${VLLM_VENV}" uv sync \
    --locked \
    --extra vllm \
    --no-install-package deep-ep \
    --no-install-package deep-gemm \
    --directory "${NEMO_RL_ROOT}"
fi

if [[ -x "${AUTOMODEL_VENV}/bin/python" ]] && \
  "${AUTOMODEL_VENV}/bin/python" -c \
    "import torch, nemo_automodel, flash_attn" >/dev/null 2>&1; then
  echo "Reusing ready dense AutoModel policy environment."
else
  echo "Preparing dense AutoModel policy environment..."
  uv venv --allow-existing "${AUTOMODEL_VENV}"
  UV_PROJECT_ENVIRONMENT="${AUTOMODEL_VENV}" uv sync \
    --locked \
    --extra automodel \
    --no-install-package deep-ep \
    --no-install-package mamba-ssm \
    --no-install-package causal-conv1d \
    --no-install-package nv-grouped-gemm \
    --no-install-package transformer-engine \
    --no-install-package transformer-engine-torch \
    --directory "${NEMO_RL_ROOT}"
fi

"${VLLM_VENV}/bin/python" -c \
  'from transformers import AutoConfig; AutoConfig.for_model("gemma4"); import torch, vllm; print(f"vLLM worker ready: torch={torch.__version__}, vllm={vllm.__version__}")'
"${AUTOMODEL_VENV}/bin/python" -c \
  "import torch, nemo_automodel; print(f'AutoModel worker ready: torch={torch.__version__}')"

echo "Dense worker environments are ready."
