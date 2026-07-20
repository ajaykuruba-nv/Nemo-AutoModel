# EkaCare Clinical Note GRPO with NeMo RL and NeMo Gym

This project fine-tunes `google/gemma-4-E4B-it` on the gated
[`ekacare/clinical_note_generation_dataset`](https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset)
using Group Relative Policy Optimization (GRPO) from
[`NVIDIA-NeMo/RL`](https://github.com/NVIDIA-NeMo/RL). NeMo Gym executes the
clinical-note task and provides rubric-based rewards.

> The requested algorithm was written as “GPRO.” NeMo RL implements **GRPO**,
> so this project uses GRPO.

The recipe targets one node with **8 NVIDIA A100 GPUs**. Native `uv run` is the
only execution method documented here. Docker is not used because the latest
released NGC image, `nvcr.io/nvidia/nemo-rl:v0.6.0`, predates Gemma 4 support.

## Project contents

```text
Ekacare/
├── configs/
│   ├── clinical_note_gym.yaml          # Clinical NeMo Gym agent and judge
│   └── grpo_gemma4_e4b_8xa100.yaml     # Eight-GPU GRPO configuration
├── scripts/
│   ├── prepare_dataset.py              # Hugging Face dataset -> Gym JSONL
│   ├── prepare_data.sh                 # Dataset preparation entry point
│   ├── native_preflight.sh             # uv, Gemma 4, CUDA, worker validation
│   ├── bootstrap_dense_envs.sh         # Dense AutoModel/vLLM worker setup
│   ├── run_grpo_dense.py               # NeMo RL/Gym launcher integration
│   └── train.sh                         # Native uv training entry point
├── tests/test_prepare_dataset.py
├── .env.example
├── .gitignore
└── requirements-data.txt
```

Generated datasets, credentials, model caches, logs, and checkpoints are not
committed.

## Implementation details

### Dataset conversion

The source dataset contains 156 examples in its `test` split with the fields
`session_id`, `text`, `text_md5`, `sample_prompt`, and `rubrics`. Because it has
no published training split, preparation creates a deterministic local split:

- 140 training examples;
- 16 validation examples;
- shuffle seed `42`.

`scripts/prepare_dataset.py` uses `sample_prompt` as the policy prompt and falls
back to a prompt built from `text`. It converts JSON, list, dictionary, and
line-oriented rubric representations into weighted NeMo Gym criteria:

```json
{
  "question": "The note preserves the clinically important negation.",
  "pass_criteria": "YES",
  "weight": 1.0
}
```

Each prepared record contains the medical-scribe instruction, policy prompt,
source conversation for judge context, normalized rubric criteria, and the
`clinical_note_simple_agent` Gym route.

### GRPO and reward path

```text
EkaCare prompt
    -> Gemma 4 produces four candidate clinical notes
    -> NeMo Gym simple agent
    -> clinical rubric resource server
    -> one YES/NO decision per criterion
    -> weighted reward in [0, 1]
    -> group-relative advantages
    -> GRPO policy update
```

The colocated Gemma vLLM endpoint generates notes and judges rubric criteria,
keeping the experiment within eight GPUs. This self-judging reward is useful
for experimentation but is non-stationary. Production evaluation should use a
frozen, independently validated clinical judge.

Each prompt receives four completions. Rewards are normalized within the group,
and a reference-policy KL coefficient of `0.01` limits drift.

### Model and GPU configuration

| Setting | Value |
|---|---:|
| Model | `google/gemma-4-E4B-it` |
| Precision | BF16 |
| Hardware | 1 node × 8 A100 GPUs |
| Training backend | NeMo AutoModel/DTensor |
| Generation backend | vLLM |
| Training tensor parallelism | 1 |
| vLLM tensor parallelism | 4 |
| Prompts per step | 8 |
| Generations per prompt | 4 |
| Global training batch | 32 |
| Maximum total sequence | 4096 tokens |
| Maximum generated note | 2048 tokens |
| Attention/linear backend | PyTorch SDPA / PyTorch linear |
| Optimizer | PyTorch AdamW |
| Learning rate | `1e-6` |
| Activation checkpointing | enabled |

Gemma 4 is multimodal, but this dataset is text-only. The vision and audio
towers are frozen while the language model is optimized.

### Why NeMo RL v0.6.0 is not used

The v0.6.0 container starts Ray and the vLLM actors, but its isolated worker
environment fails with:

```text
KeyError: 'gemma4'
Transformers does not recognize this architecture
```

Gemma 4 support landed after v0.6.0 in NeMo RL commit `911dfc31d` and included
coordinated NeMo RL, NeMo AutoModel, Transformers, checkpointing, training, and
vLLM changes. Upgrading only Transformers in v0.6.0 is therefore not a valid
fix.

This project uses the source checkout at:

```text
/userhome/home/akumarkuruba/rl/nemo-rl
```

The checkout used during implementation was commit
`80e86309b4ba115adb5a5576b3976f8a4bf5fd01`, which contains Gemma 4 support.

### Dense worker environments

NeMo RL gives vLLM and AutoModel workers isolated `uv` environments. Its broad
extras include optional DeepEP, DeepGEMM, Mamba, grouped-GEMM, and Transformer
Engine extensions. This dense recipe uses PyTorch SDPA, PyTorch linear layers,
and AdamW, so `scripts/bootstrap_dense_envs.sh` creates only the required locked
worker environments while omitting unused MoE/Mamba/Transformer Engine source
packages.

`scripts/run_grpo_dense.py` routes Ray actors to those prebuilt interpreters,
avoiding a broad environment rebuild each time Ray starts.

## Prerequisites

- Linux with eight NVIDIA A100 GPUs.
- Git and [`uv`](https://docs.astral.sh/uv/).
- The NeMo RL source checkout with all submodules.
- At least 100 GB free for dependencies, Gemma, caches, and checkpoints.
- Accepted Hugging Face access for:
  - <https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset>
  - <https://huggingface.co/google/gemma-4-E4B-it>
- A replacement Hugging Face read token in `HF_TOKEN` or `.env`.
- A CUDA-13-capable NVIDIA driver, or NVIDIA's CUDA 13.2 forward-compatibility
  package installed by the node administrator.

The token originally supplied in chat must be considered exposed. Revoke it
and use a replacement token. Never place a real token in source, YAML, or shell
command arguments.

## 1. Prepare the NeMo RL checkout

The expected layout is:

```text
/userhome/home/akumarkuruba/
├── Ekacare/
└── rl/nemo-rl/
```

If the checkout already exists, initialize its pinned submodules:

```bash
cd /userhome/home/akumarkuruba/rl/nemo-rl
git submodule update --init --recursive
```

Confirm that it contains the Gemma 4 support commit:

```bash
git merge-base --is-ancestor 911dfc31d HEAD \
  && echo "Gemma 4 NeMo RL support is present"
```

Install the locked main environment and NeMo Gym integration:

```bash
uv sync --locked --extra nemo_gym
```

Do not install arbitrary Torch, Transformers, vLLM, or AutoModel versions with
`pip`. The NeMo RL lockfile is the dependency authority.

## 2. Resolve the native CUDA compatibility requirement

The selected NeMo RL checkout uses Python 3.13.13 and CUDA 13 PyTorch/vLLM
wheels. This host's current driver advertises CUDA 12.8, which caused the
previous native run to fail during `torch.cuda` initialization.

On A100 data-center GPUs, NVIDIA supports a forward-compatibility package. Ask
the node administrator to install the package matching this checkout:

```bash
sudo apt-get install cuda-compat-13-2
```

The expected library after installation is:

```text
/usr/local/cuda-13.2/compat/libcuda.so.1
```

The project launcher automatically prepends `/usr/local/cuda-13.2/compat` to
`LD_LIBRARY_PATH` when that file exists. If the administrator installs it
elsewhere, specify the directory:

```bash
export CUDA_COMPAT_DIR=/approved/path/to/cuda-13.2/compat
```

An administrator can instead upgrade the host driver to one supporting the
checkout's CUDA version directly. Do not copy random `libcuda.so` files from
another system: user-mode driver libraries must be compatible with the host
kernel driver.

## 3. Configure Hugging Face credentials

```bash
cd /userhome/home/akumarkuruba/Ekacare
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```text
HF_TOKEN=hf_your_new_read_token
```

An exported `HF_TOKEN` takes precedence over `.env`. The account owning the
token must have accepted both gated repositories. Fine-grained tokens must
explicitly include read access to both repositories.

## 4. Prepare the dataset

```bash
cd /userhome/home/akumarkuruba/Ekacare
./scripts/prepare_data.sh
```

Expected summary:

```json
{
  "dataset_id": "ekacare/clinical_note_generation_dataset",
  "source_split": "test",
  "seed": 42,
  "train_samples": 140,
  "validation_samples": 16
}
```

Generated files:

```text
data/train.jsonl
data/validation.jsonl
data/manifest.json
```

To change the split:

```bash
./scripts/prepare_data.sh --validation-size 20 --seed 123
```

## 5. Run the native preflight

The preflight validates the main locked environment, Gemma 4 registration,
CUDA initialization on exactly eight GPUs, and both isolated worker
environments:

```bash
cd /userhome/home/akumarkuruba/Ekacare
./scripts/native_preflight.sh
```

A successful result ends with:

```text
Main environment ready: ... GPUs=8, GPU 0=NVIDIA A100...
vLLM worker ready: ...
Native NeMo RL preflight passed.
```

The first worker build can take several minutes and requires access to package
registries and GitHub. Later runs reuse the environments.

## 6. Run two GRPO iterations

```bash
cd /userhome/home/akumarkuruba/Ekacare

./scripts/train.sh \
  grpo.max_num_steps=2 \
  grpo.max_num_epochs=1 \
  grpo.val_at_start=false \
  grpo.val_at_end=true \
  grpo.val_period=100 \
  checkpointing.save_period=1 \
  logger.wandb_enabled=false
```

This is a real eight-GPU run. Startup can take longer than the two optimization
steps because Gemma, AutoModel, vLLM, Ray, and NeMo Gym must initialize.

A healthy run should show:

- Ray discovering one node and eight GPUs;
- vLLM starting its OpenAI-compatible endpoint;
- the Gym head, agent, and clinical rubric server becoming ready;
- non-identical rewards in at least some four-completion groups;
- training steps 1 and 2 completing;
- final validation metrics;
- checkpoints under `results/gemma4-e4b-clinical-grpo`.

## 7. Run the configured experiment

After the smoke test passes:

```bash
./scripts/train.sh
```

The default recipe runs up to three passes over the 140-example training split,
validates every five updates, and keeps the best three checkpoints.

Useful overrides:

```bash
# One epoch
./scripts/train.sh grpo.max_num_epochs=1

# Lower rollout volume
./scripts/train.sh \
  grpo.num_prompts_per_step=4 \
  policy.train_global_batch_size=16

# Enable Weights & Biases
./scripts/train.sh logger.wandb_enabled=true
```

Maintain:

```text
policy.train_global_batch_size
  = grpo.num_prompts_per_step × grpo.num_generations_per_prompt
```

## Outputs and resuming

| Output | Location |
|---|---|
| Prepared data | `data/` |
| Hugging Face/Gym caches | `.cache/` |
| NeMo RL/Gym logs | `logs/gemma4-e4b-clinical-grpo*` |
| Checkpoints | `results/gemma4-e4b-clinical-grpo/` |
| TensorBoard events | Under the experiment log directory |

Re-running the same command resumes the latest compatible checkpoint. For a
separate experiment, override both destinations:

```bash
./scripts/train.sh \
  checkpointing.checkpoint_dir=/userhome/home/akumarkuruba/Ekacare/results/experiment-2 \
  logger.log_dir=/userhome/home/akumarkuruba/Ekacare/logs/experiment-2
```

## Troubleshooting

### CUDA reports that the driver is too old

Run:

```bash
ls -l /usr/local/cuda-13.2/compat/libcuda.so.1
CUDA_COMPAT_DIR=/usr/local/cuda-13.2/compat ./scripts/native_preflight.sh
```

If the library is absent, the administrator must install
`cuda-compat-13-2` or upgrade the host driver. If it exists but initialization
still fails, verify that the compatibility directory is first in
`LD_LIBRARY_PATH` and ask the administrator to validate the driver/package
combination.

### Worker setup fails in `create_local_venv`, DeepEP, or Mamba

Always launch with this project's scripts. To retry the two required dense
worker environments explicitly:

```bash
./scripts/bootstrap_dense_envs.sh
```

The launcher intentionally omits unused MoE/Mamba/Transformer Engine source
extensions. A DNS/download failure still requires outbound package access or a
populated `uv` cache.

### `KeyError: 'gemma4'`

This means an old environment is being used. Confirm the NeMo RL checkout
contains commit `911dfc31d`, then rerun:

```bash
./scripts/native_preflight.sh
```

Do not repair the v0.6.0 container using only a Transformers upgrade.

### Hugging Face returns `401`, `403`, or a gated-repository error

- Replace revoked or placeholder credentials in `.env`.
- Use `unset HF_TOKEN` if the shell exports an older token that masks `.env`.
- Accept access for both repositories with the token owner's account.
- Explicitly grant both repositories to a fine-grained token.

Never print the complete token in diagnostic logs.

### CUDA out of memory

```bash
./scripts/train.sh \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.4 \
  grpo.num_prompts_per_step=4 \
  policy.train_global_batch_size=16
```

Keep four generations per prompt so GRPO retains a useful group baseline. If
needed, reduce `policy.max_total_sequence_length` and
`policy.generation.max_new_tokens` together.

### Every reward in a group is identical

GRPO cannot learn from a group with zero reward variance. Inspect Gym responses
and confirm rubric items are non-empty and judge outputs terminate with
`[[YES]]` or `[[NO]]`. A frozen external clinical judge is the more reliable
long-term design.

### Port conflict

NeMo RL uses ports `11001-15000` for vLLM, and this recipe reserves
`15001-20000` for NeMo Gym. Stop only stale processes belonging to this
experiment; do not indiscriminately terminate processes on a shared node.

## Important limitations

1. **Self-judging reward:** the policy endpoint also judges rubric criteria,
   creating non-stationary rewards and reward-hacking risk.
2. **Small dataset:** 156 examples cannot establish broad clinical quality,
   safety, or generalization.
3. **No clinical deployment claim:** reward does not demonstrate that notes are
   suitable for patient care. Qualified clinician review is required.
4. **Sensitive outputs:** protect conversations, generated notes, logs,
   checkpoints, and external experiment traces.
5. **Derived validation split:** do not report the local validation result as
   performance on an untouched official test set.

## References

- [NeMo RL repository](https://github.com/NVIDIA-NeMo/RL)
- [NeMo RL dependency management](https://docs.nvidia.com/nemo/rl/latest/design-docs/dependency-management.html)
- [NeMo RL GRPO guide](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html)
- [NeMo Gym repository](https://github.com/NVIDIA-NeMo/Gym)
- [NeMo Gym with NeMo RL](https://docs.nvidia.com/nemo/gym/latest/training-tutorials/nemo-rl-grpo/nemo-rl-configuration.html)
- [NVIDIA CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
- [EkaCare clinical-note dataset](https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset)
- [Gemma 4 E4B Instruct](https://huggingface.co/google/gemma-4-E4B-it)
