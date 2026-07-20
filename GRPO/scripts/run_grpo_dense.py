#!/usr/bin/env python3
"""Launch upstream NeMo Gym GRPO with prebuilt dense-only worker venvs."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def require_python(path: Path, label: str) -> str:
    python = path / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(
            f"{label} worker environment is missing at {python}. "
            "Run scripts/bootstrap_dense_envs.sh first."
        )
    return str(python)


def main() -> None:
    nemo_root = Path(
        os.environ.get("NEMO_RL_ROOT", "/userhome/home/akumarkuruba/rl/nemo-rl")
    ).resolve()
    venv_root = Path(os.environ.get("NEMO_RL_VENV_DIR", nemo_root / "venvs"))
    vllm_python = require_python(
        venv_root
        / "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        "vLLM",
    )
    automodel_python = require_python(
        venv_root
        / "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2",
        "AutoModel",
    )

    # Patch the executable constants before the actor registry is materialized.
    # Direct Python paths bypass NeMo RL's broad `uv run --extra ...` resync,
    # which would attempt unused CUDA source builds on every launch.
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    PY_EXECUTABLES.VLLM = vllm_python
    PY_EXECUTABLES.AUTOMODEL = automodel_python

    from nemo_rl.distributed import ray_actor_environment_registry as registry

    vllm_actor_names = {
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
        "nemo_rl.algorithms.async_utils.ReplayBuffer",
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor",
    }
    automodel_actor_names = {
        "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker",
        "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2",
        "nemo_rl.models.value.workers.dtensor_value_worker_v2.DTensorValueWorkerV2",
    }
    for name in vllm_actor_names:
        if name in registry.ACTOR_ENVIRONMENT_REGISTRY:
            registry.ACTOR_ENVIRONMENT_REGISTRY[name] = vllm_python
    for name in automodel_actor_names:
        if name in registry.ACTOR_ENVIRONMENT_REGISTRY:
            registry.ACTOR_ENVIRONMENT_REGISTRY[name] = automodel_python

    upstream = nemo_root / "examples" / "nemo_gym" / "run_grpo_nemo_gym.py"
    if not upstream.is_file():
        raise RuntimeError(f"Upstream NeMo Gym launcher not found: {upstream}")
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()

