#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-/userhome/home/akumarkuruba/rl/nemo-rl}"

# Load the project-local environment automatically when the caller has not
# already supplied a token. `set -a` also exports assignments that do not use
# the optional `export` prefix.
if [[ -z "${HF_TOKEN:-}" && -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set." >&2
  echo "Add it to ${PROJECT_ROOT}/.env or export it in this shell." >&2
  echo "Also accept the dataset access terms in the same Hugging Face account." >&2
  exit 2
fi

if [[ "${HF_TOKEN}" == "hf_your_token_here" ]]; then
  echo "HF_TOKEN still contains the placeholder from .env.example." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
uv run --project "${NEMO_RL_ROOT}" --with-requirements requirements-data.txt \
  python scripts/prepare_dataset.py "$@"
