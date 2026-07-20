# Lecture Notes: Understanding The Ekacare Clinical Note GRPO Implementation

## 1. The Project At A High Level

The Ekacare folder is a compact but complete experiment recipe for fine-tuning a language model to generate clinical notes from clinician-patient conversations. The model chosen by this project is `google/gemma-4-E4B-it`, and the optimization method is Group Relative Policy Optimization, usually abbreviated as GRPO. The surrounding training infrastructure comes from NVIDIA NeMo RL, while the reward evaluation path is delegated to NeMo Gym.

The project is not a general web application or a package with many internal modules. It is better understood as an experiment harness. It prepares a gated Hugging Face dataset, converts it into the JSONL format expected by NeMo Gym, configures the policy model, configures the reward judge, validates the native GPU environment, builds worker environments for the distributed actors, and finally launches the upstream NeMo RL GRPO training loop.

The main flow is:

```text
Hugging Face clinical conversation dataset
        -> scripts/prepare_dataset.py
        -> data/train.jsonl and data/validation.jsonl
        -> configs/grpo_gemma4_e4b_8xa100.yaml
        -> scripts/train.sh
        -> scripts/run_grpo_dense.py
        -> upstream NeMo RL GRPO launcher
        -> Gemma policy generates notes
        -> NeMo Gym judge scores notes using rubrics
        -> GRPO updates the policy
```

The important intuition is that this repository does not reimplement GRPO itself. Instead, it supplies the data, configuration, environment bootstrapping, and launcher glue needed to make the upstream NeMo RL implementation train Gemma 4 on a clinical-note task.

## 2. The Learning Problem

The underlying task is clinical note generation. Each raw dataset example contains a medical conversation and a prompt asking for a clinical note. It also contains rubric items describing what a good note should satisfy. The model sees the instruction and conversation-derived prompt, produces a clinical note, and receives a reward based on whether the note satisfies the rubric.

Mathematically, we can describe the task as conditional generation. Let \(x\) be the prompt, which includes the clinical context and instruction. Let \(y = (y_1, y_2, \ldots, y_T)\) be the generated clinical note token sequence. The policy model is an autoregressive distribution:

```text
pi_theta(y | x) = product over t from 1 to T of pi_theta(y_t | x, y_<t)
```

Here, \(\theta\) represents the trainable parameters of Gemma's language model. Fine-tuning means modifying \(\theta\) so that high-quality clinical notes become more likely and low-quality notes become less likely.

The reward is not a single human-written score. It is computed by judging the candidate note against a set of rubric criteria. If a row has rubric criteria \(c_1, c_2, \ldots, c_m\), and each criterion has weight \(w_i\), the judge produces a binary verdict for each criterion. Let \(z_i = 1\) if the candidate satisfies criterion \(c_i\), and \(z_i = 0\) otherwise. A natural weighted reward is:

```text
R(x, y) = (sum over i of w_i z_i) / (sum over i of w_i)
```

This explains why the repository normalizes rubrics into question, pass criterion, and weight fields. The training algorithm needs a scalar reward, but the clinical task is naturally multi-criterion. Rubric normalization bridges those two worlds.

## 3. Repository Structure

The `configs` directory contains the declarative experiment definitions. `configs/grpo_gemma4_e4b_8xa100.yaml` configures GRPO, Gemma 4, vLLM generation, NeMo Gym, data paths, checkpointing, logging, and the eight-A100 cluster shape. `configs/clinical_note_gym.yaml` defines the NeMo Gym clinical-note resource server and the simple agent route used by each prepared dataset record.

The `scripts` directory contains executable entry points. `scripts/prepare_dataset.py` is the actual data conversion program. `scripts/prepare_data.sh` wraps that converter with Hugging Face token loading and NeMo RL project execution. `scripts/native_preflight.sh` checks that the local environment can see Gemma 4, CUDA, and the required worker interpreters. `scripts/bootstrap_dense_envs.sh` builds the smaller worker environments needed for this dense model recipe. `scripts/run_grpo_dense.py` patches NeMo RL's actor interpreter registry before handing control to the upstream GRPO launcher. `scripts/train.sh` is the main user-facing training entry point.

The `tests` directory contains focused tests for dataset preparation. `tests/test_prepare_dataset.py` verifies the two most important local behaviors: rubrics are normalized correctly, and converted examples point to the expected NeMo Gym agent.

The `data` directory is expected to hold generated JSONL data. The committed `data/manifest.json` records that the current prepared split comes from the gated EkaCare dataset's `test` split, with seed `42`, producing `140` training examples and `16` validation examples.

## 4. Dataset Preparation From First Principles

The data conversion program lives in `scripts/prepare_dataset.py`. Its purpose is to translate examples from the source dataset into the structure expected by NeMo Gym and NeMo RL.

The source dataset is described by the README as having fields such as `session_id`, `text`, `text_md5`, `sample_prompt`, and `rubrics`. The model should not receive arbitrary raw rows. The training system expects each row to include an agent reference, message-style model input, rubric information, context for the judge, and metadata. The converter constructs exactly that.

The constant `DEFAULT_DATASET` names the Hugging Face dataset: `ekacare/clinical_note_generation_dataset`. The constant `DEFAULT_SYSTEM_PROMPT` is the system instruction given to the policy. Its role is important. It constrains the model toward faithful medical scribing: include only supported facts, preserve negations and uncertainty, and do not invent diagnoses, medications, tests, results, or advice.

This is not only natural-language decoration. In language-model training, the prompt is part of the conditioning variable \(x\). A better system prompt changes the conditional distribution the model learns from and the distribution it samples during rollouts. If the prompt tells the model to avoid unsupported facts, then the reward judge and the policy instruction point in the same direction.

## 5. Rubric Normalization

The function `normalize_rubrics` is one of the most important pieces of local implementation. Clinical rubrics may arrive as JSON strings, Python-style lists after dataset loading, dictionaries, plain strings, or line-oriented text. NeMo Gym, however, needs each item in a stable shape:

```json
{
  "question": "The note preserves the clinically important negation.",
  "pass_criteria": "YES",
  "weight": 1.0
}
```

The helper `_maybe_json` first checks whether a value is a string that looks like JSON. It trims whitespace and only attempts `json.loads` when the first character suggests JSON-like structure. This is a defensive parsing strategy. If every string were blindly parsed as JSON, ordinary rubric text could throw unnecessary decoding errors. If no string were parsed, serialized rubric arrays would remain opaque text and the judge would not receive individual criteria.

The normalization logic handles dictionaries by looking for keys such as `rubrics`, `rubric`, `criteria`, and `items`. If one exists, that nested value becomes the real rubric list. If none exists, the dictionary's values are used. This makes the converter robust to multiple possible dataset encodings.

If the rubric is a plain string, the code splits it into lines and removes common list markers such as hyphens, asterisks, `1.`, or `1)`. In regular-expression terms, the pattern strips leading whitespace followed by a bullet or numbered-list marker. After stripping, empty lines are discarded. This allows a rubric written as a human-readable checklist to become machine-readable criteria.

For each final rubric item, the converter extracts the question text, expected pass criterion, and weight. Strings become a criterion with default pass criterion `YES` and weight `1.0`. Dictionaries can specify the question using several possible key names: `question`, `rubric`, `criterion`, `criteria`, `description`, or `text`. This matters because different datasets often use slightly different schema names for the same semantic field.

The mathematical reason for keeping weights is that not every criterion must contribute equally. If criterion \(i\) has weight \(w_i\), then its contribution to total reward is proportional to \(w_i\). A clinically critical negation, such as "no fever" versus "fever", could be assigned greater weight than a formatting preference. In the current conversion code, missing weights default to `1.0`, which makes the reward an ordinary average over satisfied criteria.

## 6. Prompt Construction

The function `build_prompt` chooses the policy prompt. It first tries to use `sample_prompt`. If that field is present and non-empty, it is assumed to be the dataset-provided task prompt. If not, the function falls back to building a generic prompt from the raw conversation text:

```text
Create a structured clinical note from this conversation. Return only the clinical note.

CLINICAL CONVERSATION:
...
```

This fallback is an important reliability feature. Training data pipelines often fail because one optional field is missing. Here, the implementation says that the raw conversation is enough to create a usable prompt. The resulting behavior is conservative: it does not invent a new task, it simply asks for the core dataset objective.

In terms of the learning setup, `build_prompt` constructs \(x\), the conditioning input. If \(x\) is empty, there is no meaningful task instance. That is why `convert_row` raises an error when neither prompt nor text is available.

## 7. Converting A Row Into A NeMo Gym Example

The function `convert_row` is where a raw dataset row becomes a training example. It calls `build_prompt`, normalizes the rubrics, validates both, chooses a session identifier, and returns a dictionary with the exact structure consumed downstream.

The `uuid` field is the stable example identifier. If the dataset row has a `session_id`, that becomes the UUID. Otherwise, the converter creates a deterministic fallback such as `row-0000`. The `task_id` is the numeric index in the loaded row list.

The `agent_ref` field is central:

```json
{
  "type": "responses_api_agents",
  "name": "clinical_note_simple_agent"
}
```

This tells NeMo Gym which agent configuration should process the example. The name must match a configured agent in `configs/clinical_note_gym.yaml`. In this project, that agent is `clinical_note_simple_agent`.

The `responses_create_params` field contains model input in a chat-style format. It includes the medical-scribe system message and the user prompt. During rollout, the policy model receives this input and generates a clinical note.

The `rubric` field contains normalized criteria. The `context` field contains the source clinical conversation or prompt, and it is especially important for judging. The judge cannot reliably decide whether a note invented facts unless it can compare the candidate note against the original conversation. The `metadata` field carries traceability information such as session ID, MD5 of source text, and source dataset name.

The key design idea is separation of roles. The policy receives instructions and produces a note. The judge receives the original context, generated note, and rubric criterion, then emits a binary judgment. The same JSONL record contains enough information for both roles.

## 8. Loading Rows And Credential Safety

The function `load_rows` supports two data sources. If `--input-jsonl` is provided, it reads local JSONL. This is useful for offline testing and development because it avoids a network call and gated Hugging Face access. If no local input is provided, it loads from Hugging Face.

Before accessing Hugging Face, the function requires either `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`. It explicitly rejects an empty token and the placeholder token `hf_your_token_here`. This is a safety-oriented implementation detail. A placeholder token would produce confusing authentication errors, and accidentally embedding real credentials into source-controlled commands would be dangerous.

The function then calls `HfApi().whoami(token=token)` to validate the token and prints the authenticated account name. This gives the user actionable feedback when gated repository permissions fail. Then it calls `load_dataset(dataset_id, split=split, token=token)`. If Hugging Face reports that the dataset cannot be found, the code raises a more helpful error explaining that the authenticated account may not have accepted access terms or may need fine-grained repository permission.

This is a common pattern in robust ML infrastructure. Raw library exceptions are often technically correct but operationally unhelpful. The wrapper converts them into explanations that match the actual failure modes of gated datasets.

## 9. Deterministic Splitting

The `main` function parses command-line arguments, loads rows, optionally truncates them with `--max-samples`, validates the dataset size, converts every row, shuffles the converted rows using `random.Random(args.seed).shuffle(converted)`, and then splits the shuffled list into validation and training sets.

The validation set is the first `validation_size` examples after shuffling, and the training set is the rest. With the defaults, this means `16` validation examples and `140` training examples. Because the random generator is seeded, the split is deterministic. Anyone using the same input rows, seed, and validation size should get the same split.

Determinism matters because model training already contains many stochastic components. If the data split changes silently, it becomes harder to compare experiments. The seed makes this part of the experiment reproducible.

Let \(D\) be the converted dataset of \(n\) examples. A seeded shuffle applies a deterministic permutation \(\sigma\) to the examples. The validation set is:

```text
V = { D_{sigma(1)}, D_{sigma(2)}, ..., D_{sigma(k)} }
```

and the training set is:

```text
T = { D_{sigma(k+1)}, ..., D_{sigma(n)} }
```

where \(k\) is `validation_size`. For this project, \(n = 156\) and \(k = 16\), so \(|T| = 140\).

## 10. JSONL Output Format

The helper `write_jsonl` writes one JSON object per line. JSONL is widely used in training pipelines because it is stream-friendly. A training process can read examples line by line without loading an entire dataset into memory. It is also easy to concatenate, shard, inspect, and validate.

The converter writes `data/train.jsonl`, `data/validation.jsonl`, and `data/manifest.json`. The manifest records the dataset ID, source split, seed, and sample counts. This metadata is small, but it is valuable because it tells future readers how the local files were derived.

A typical prepared example contains a UUID, task ID, agent reference, model input, rubric, context, and metadata. During training, the data loader reads these examples and passes them into the NeMo Gym environment through the `nemo_gym_data_processor` specified in the YAML config.

## 11. Shell Entry Point For Data Preparation

The script `scripts/prepare_data.sh` is a thin operational wrapper around `prepare_dataset.py`. It computes the project root from its own path, sets a default NeMo RL root, loads `.env` only if `HF_TOKEN` is not already exported, rejects missing or placeholder tokens, and then runs:

```bash
uv run --project "${NEMO_RL_ROOT}" --with-requirements requirements-data.txt python scripts/prepare_dataset.py "$@"
```

The use of `uv run --project "${NEMO_RL_ROOT}"` means the converter executes in the dependency context of the NeMo RL checkout rather than a random system Python. The additional `--with-requirements requirements-data.txt` supplies the dataset-specific libraries, namely `datasets` and `huggingface-hub`.

The wrapper also preserves command-line flexibility. Any arguments passed to `prepare_data.sh`, such as `--validation-size 20 --seed 123`, are forwarded directly to the Python converter.

## 12. The GRPO Configuration

The file `configs/grpo_gemma4_e4b_8xa100.yaml` is the central experiment specification. It begins by inheriting defaults from an upstream NeMo RL config:

```yaml
defaults: /userhome/home/akumarkuruba/rl/nemo-rl/examples/configs/grpo_math_1B.yaml
```

The local file then overrides the parts needed for the EkaCare clinical-note experiment. Conceptually, this is configuration inheritance. The upstream file supplies generic GRPO machinery, and the local file changes the model, data, reward environment, logging, checkpointing, and hardware layout.

The `grpo` section sets rollout and training-level parameters. It uses `8` prompts per step and `4` generations per prompt. Therefore, each optimization step considers:

```text
8 prompts x 4 generations per prompt = 32 generated completions
```

This matches `policy.train_global_batch_size: 32`. The equality is important. GRPO computes relative advantages within groups of generations for the same prompt. If the policy batch size did not match the rollout structure, training would be poorly aligned or fail depending on upstream validation.

The configuration uses `max_rollout_turns: 1`, which fits clinical note generation. This is not a multi-turn agentic task. The model receives a prompt and produces one answer. The experiment runs up to three epochs by default, validates every five updates, validates at the start and end, normalizes rewards, and uses a leave-one-out baseline.

## 13. GRPO Intuition

GRPO is a reinforcement-learning method for language models that compares multiple completions for the same prompt. Instead of needing a separate learned value function for every state, it samples a group of outputs, scores them, and uses relative reward within the group to decide which outputs should become more likely.

For a prompt \(x\), suppose the model samples \(G\) completions:

```text
y_1, y_2, ..., y_G ~ pi_theta(. | x)
```

In this project, \(G = 4\). The reward model or environment gives scalar rewards:

```text
r_1, r_2, ..., r_G
```

The simplest group-relative advantage for completion \(j\) is:

```text
A_j = r_j - mean(r_1, r_2, ..., r_G)
```

With reward normalization, the advantage is often scaled by the group standard deviation:

```text
A_j = (r_j - mean(r)) / (std(r) + epsilon)
```

The configured leave-one-out baseline uses a closely related idea. For completion \(j\), the baseline excludes \(r_j\):

```text
b_j = (sum over k != j of r_k) / (G - 1)
```

and the advantage becomes:

```text
A_j = r_j - b_j
```

The intuition is simple. If four notes are generated for the same conversation, and one note satisfies more rubric criteria than the other three, the good note receives a positive advantage. If another note misses important clinical facts, it receives a negative advantage. The model update then increases the probability of token sequences from better-than-peer notes and decreases the probability of worse-than-peer notes.

This group-relative design is especially helpful when absolute reward calibration is imperfect. The judge may be noisy, and the scale of rewards may vary across prompts. Comparing completions under the same prompt reduces some of that variation because all completions face the same conversation and rubric.

## 14. Policy Gradient View

The policy-gradient objective tries to increase expected reward:

```text
J(theta) = E_{x ~ D, y ~ pi_theta(. | x)} [ R(x, y) ]
```

The classic score-function gradient is:

```text
grad_theta J(theta) = E [ R(x, y) grad_theta log pi_theta(y | x) ]
```

A baseline can be subtracted without changing the expected gradient:

```text
grad_theta J(theta) = E [ (R(x, y) - b(x)) grad_theta log pi_theta(y | x) ]
```

GRPO uses group rewards to form that baseline. Since:

```text
log pi_theta(y | x) = sum over t of log pi_theta(y_t | x, y_<t)
```

the token-level update distributes the completion-level advantage over generated tokens. The YAML sets `token_level_loss: true`, meaning loss is computed at token granularity rather than treating the sequence as an indivisible unit.

There is also a reference-policy KL penalty:

```text
loss_fn.reference_policy_kl_penalty: 0.01
```

The KL term discourages the trained policy from drifting too far from a reference model. Informally, the optimized objective becomes:

```text
maximize E[ A(y) log pi_theta(y | x) ] - beta KL(pi_theta || pi_ref)
```

where \(\beta = 0.01\). This is important in clinical generation because reward optimization can otherwise push the model toward strange outputs that exploit the judge rather than genuinely improving notes.

## 15. Why Reward Variance Matters

GRPO learns from differences within a group. If all four generated notes for a prompt receive the same reward, then the group-relative advantages are zero or nearly zero. For example, if:

```text
r = [0.75, 0.75, 0.75, 0.75]
```

then every completion is exactly average. The algorithm receives no preference signal for that prompt. This is why the README warns that every reward in a group being identical is a problem. In a healthy run, at least some prompts should produce non-identical rewards across the four sampled completions.

Sampling temperature contributes to this. The configuration uses `temperature: 1.0`, `top_p: 1.0`, and `top_k: null`, which allows diverse completions. Diversity is useful because GRPO needs alternatives to compare. If generation were nearly deterministic, all four outputs could be too similar, reducing the available learning signal.

## 16. The Clinical Note Gym Configuration

The file `configs/clinical_note_gym.yaml` defines two named components: `clinical_note` and `clinical_note_simple_agent`.

The `clinical_note` component is a resource server using Gym's maintained multichallenge-style rubric server. It is connected to the same colocated policy model server as the judge. The judge prompt template includes the original clinical conversation, the candidate clinical note, the criterion, and the expected verdict. The judge is instructed to end with exactly `[[YES]]` or `[[NO]]`.

This exact label protocol matters. The reward server needs to parse a judge response into a binary result. Free-form explanations would be ambiguous. By requiring a final label, the system turns language-model evaluation back into structured reward data.

The aggregation mode is `weighted`, so the rubric item weights influence the final reward. Parallel evaluation is enabled, which allows multiple rubric criteria to be judged concurrently. For a row with \(m\) criteria, parallel judging can reduce latency because each criterion is an independent binary decision conditioned on the same context and candidate response.

The `clinical_note_simple_agent` component maps the dataset's `agent_ref` to an actual executable Gym agent. It uses the policy model server to generate the answer and the clinical-note resource server to evaluate it. Its `max_steps` is `1`, confirming that the task is a single-response generation problem rather than an iterative tool-using agent task.

## 17. Self-Judging And Its Consequences

The project uses the colocated Gemma vLLM endpoint both to generate candidate notes and to judge rubric criteria. This is operationally efficient because it fits inside the eight-GPU node and avoids requiring another model server. However, it has an important learning-theoretic consequence.

The reward function is not fully stationary. As training changes the policy model, the judge model may also effectively change if the same model weights are used for both roles. In reinforcement learning, a stationary reward function means \(R(x, y)\) does not change merely because the policy changed. Here, the README correctly flags self-judging as useful for experimentation but risky for production evaluation.

The main risk is reward hacking. If the model learns outputs that make the judge say `[[YES]]` without truly improving clinical quality, the scalar reward can increase while real-world usefulness decreases. A stronger production design would freeze an independently validated clinical judge or use clinician-reviewed evaluations.

## 18. Model Configuration

The policy model is `google/gemma-4-E4B-it`, and the tokenizer uses the same name. The chat template disables thinking with:

```yaml
chat_template_kwargs:
  enable_thinking: false
```

The maximum total sequence length is `4096`, and the maximum generated note length is `2048`. This means the combined prompt and completion are bounded. If the prompt is too long, it must fit within the input budget. If the note is too long, generation stops at the configured maximum. Sequence length directly influences memory use because transformer attention has quadratic cost in sequence length in the standard formulation.

For a transformer layer with sequence length \(L\), hidden dimension \(d\), and attention heads, the attention score matrix is roughly \(L \times L\). The cost of dense self-attention scales approximately as:

```text
O(L^2 d)
```

This is why reducing `policy.max_total_sequence_length` or `policy.generation.max_new_tokens` can help with CUDA out-of-memory errors. Smaller \(L\) reduces both compute and memory pressure.

The precision is `bfloat16`. BF16 is a common training precision on A100 GPUs because it reduces memory and bandwidth cost while preserving a wider exponent range than FP16. The optimizer is AdamW with learning rate `1e-6`, weight decay `0.1`, betas `[0.9, 0.999]`, and epsilon `1e-8`.

AdamW maintains first and second moment estimates:

```text
m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2
```

After bias correction, parameters are updated using a scaled gradient plus decoupled weight decay. The small learning rate reflects the risk of destabilizing a large instruction-tuned model during reinforcement learning.

## 19. DTensor, Dense Training, And Freezing

The `dtensor_cfg` enables NeMo AutoModel's DTensor policy worker path. Tensor parallelism is set to `1`, and context parallelism is also `1`. In this recipe, the model is not split tensor-wise across the training workers through these parameters. Instead, the setup relies on the broader NeMo RL distributed execution and colocated generation arrangement.

Activation checkpointing is enabled. During ordinary backpropagation, a model stores intermediate activations so gradients can be computed. Activation memory can be large for long sequences and deep transformers. Checkpointing trades compute for memory by saving fewer activations and recomputing some forward-pass values during the backward pass.

If \(M_{act}\) is the memory required to store all activations, checkpointing reduces memory at the cost of extra computation. The exact reduction depends on checkpoint placement, but the intuition is:

```text
less stored activation memory + more recomputation = feasible larger model or sequence length
```

The backend config chooses PyTorch SDPA attention and PyTorch linear layers. It also freezes the vision and audio towers while leaving the language model trainable. This matters because Gemma 4 is multimodal, but this dataset is text-only. Training vision or audio modules on a text-only clinical-note dataset would waste memory and compute and could degrade unused capabilities.

## 20. vLLM Generation Configuration

The `generation` section uses vLLM as the backend. vLLM is optimized for high-throughput language-model serving. In this project, it generates candidate clinical notes during rollout. The vLLM tensor parallel size is `4`, meaning the generation model can be partitioned across four GPUs for serving.

The vLLM config exposes an HTTP server, uses BF16, enforces eager execution, sets maximum model length to `4096`, and uses `gpu_memory_utilization: 0.5`. That memory utilization value is conservative. It leaves room for other colocated training components on the same eight-GPU node.

The configuration sets `mm_processor_cache_gb: 0`. Since the dataset is text-only, multimodal processor caching is unnecessary. This is another example of fitting the infrastructure to the actual task instead of paying for unused model capabilities.

## 21. Data Configuration

The `data` section points training to `data/train.jsonl` and validation to `data/validation.jsonl`. The dataset name is `NemoGymDataset`, the environment name is `nemo_gym`, and the processor is `nemo_gym_data_processor`.

The maximum input sequence length is `2048`. This bounds the prompt side of the example. Since maximum total sequence length is `4096` and maximum generation length is `2048`, the intended budget is balanced: up to roughly half for input and half for output.

Shuffling is enabled for training. `num_workers` is `0`, which means data loading does not spawn separate worker processes. For a small 140-example dataset, that is sensible. Extra workers would add complexity without much benefit.

## 22. Checkpointing And Metrics

Checkpointing is enabled with output under `results/gemma4-e4b-clinical-grpo`. The configuration tracks `val:accuracy`, treats higher values as better, keeps the top three checkpoints, saves every five periods, uses `safetensors`, and saves optimizer state.

The choice of `safetensors` is common for model checkpoints because the format avoids arbitrary code execution during loading and is efficient for tensor storage. Saving optimizer state is larger, but it allows true training resume. Without optimizer state, resuming would reload model weights but lose AdamW's accumulated moment estimates.

Validation accuracy here should be interpreted carefully. The validation split is locally derived from the source `test` split because the dataset does not provide a published train split. Therefore, this validation metric is useful for experiment monitoring, but it should not be reported as performance on an untouched official benchmark.

## 23. The Training Shell Script

The main launcher is `scripts/train.sh`. It validates data presence, loads Hugging Face credentials, checks that NeMo RL exists, optionally prepends CUDA compatibility libraries, verifies CUDA initialization, builds dense worker environments, and finally launches GRPO.

The CUDA compatibility logic exists because the selected NeMo RL checkout uses CUDA 13.2 PyTorch and vLLM wheels, while the host described in the README advertised CUDA 12.8. If `/usr/local/cuda-13.2/compat/libcuda.so.1` exists, the script prepends that directory to `LD_LIBRARY_PATH`. Ray workers inherit this environment variable, so the compatibility library is visible inside distributed workers too.

The script sets `PYTHONUNBUFFERED=1`, which makes logs appear promptly instead of being delayed in buffers. It sets `TOKENIZERS_PARALLELISM=false`, avoiding tokenizer worker oversubscription warnings or thread contention. It also sets local Hugging Face cache directories under `.cache/huggingface`, which keeps downloaded artifacts inside the project area.

Before launching training, it runs a CUDA sanity check through `uv run --project "${NEMO_RL_ROOT}" --extra nemo_gym`. The check imports torch, verifies that CUDA is available, and prints the number and name of detected devices. This catches environment failures before the expensive distributed training stack starts.

## 24. Dense Worker Environment Bootstrapping

NeMo RL uses separate Python environments for different Ray actor types. The script `scripts/bootstrap_dense_envs.sh` creates two specific worker environments: one for the vLLM generation worker and one for the AutoModel DTensor policy worker.

The key implementation decision is that the script keeps the locked NeMo RL dependency graph but omits unused optional packages such as DeepEP, DeepGEMM, Mamba, grouped GEMM, and Transformer Engine. This is not a random dependency deletion. The local recipe uses dense Gemma 4 with PyTorch SDPA, PyTorch linear layers, and AdamW. The omitted packages are associated with paths this experiment does not use, and some require source builds with CUDA tooling.

The script first checks whether each worker environment already exists and can import the required libraries. If it can, the script reuses it. If not, it creates or updates the venv with `uv venv --allow-existing` and then runs `uv sync` into that environment.

This gives two benefits. First, later runs are faster because ready worker environments are reused. Second, failures happen early and explicitly rather than halfway through Ray startup, where they would be harder to diagnose.

## 25. Launcher Patching For Worker Interpreters

The file `scripts/run_grpo_dense.py` is a small but subtle launcher shim. It locates the NeMo RL root, locates the prebuilt vLLM and AutoModel worker Python interpreters, verifies that they exist, and then patches NeMo RL's executable registry before running the upstream script.

The important code path imports `PY_EXECUTABLES` from `nemo_rl.distributed.virtual_cluster` and assigns:

```python
PY_EXECUTABLES.VLLM = vllm_python
PY_EXECUTABLES.AUTOMODEL = automodel_python
```

It then imports the actor environment registry and rewrites entries for known vLLM actors and AutoModel actors. The timing matters. The script patches these values before the actor registry is materialized by the upstream launcher. If the patch happened too late, Ray actors could be created with the default broad environments instead of the dense-only environments.

Finally, it finds the upstream file:

```text
/userhome/home/akumarkuruba/rl/nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py
```

and executes it with `runpy.run_path(..., run_name="__main__")`. This means the upstream launcher behaves as if it were invoked as the main Python program, while still benefiting from the registry changes made by the local shim.

## 26. Native Preflight

The script `scripts/native_preflight.sh` checks the system before a full training run. It verifies that `uv` exists, that the NeMo RL checkout contains a `uv.lock`, that CUDA compatibility libraries are available if needed, that Transformers recognizes the `gemma4` architecture, that torch sees exactly eight GPUs, and that the dense worker environments work.

The architecture check:

```python
from transformers import AutoConfig
AutoConfig.for_model("gemma4")
```

specifically catches the failure mode where an old Transformers or NeMo RL environment does not know Gemma 4. The README mentions the error `KeyError: 'gemma4'`. This preflight turns that into an immediate compatibility check.

The eight-GPU assertion is also intentional:

```python
assert torch.cuda.device_count() == 8
```

The YAML cluster config expects one node with eight A100 GPUs. Running with fewer GPUs may cause placement, memory, or scheduling failures later. The preflight catches the mismatch early.

## 27. Tests

The test file `tests/test_prepare_dataset.py` imports `scripts/prepare_dataset.py` by path rather than as an installed package. This is appropriate for a small script-based repository. It avoids requiring a package install just to test the converter.

The first test passes a JSON-serialized rubric list containing one dictionary and one string. It asserts that normalization returns two criteria and that the string criterion receives default pass criteria `YES`. This protects the flexible parsing behavior.

The second test constructs a small raw row with a session ID, conversation text, prompt, and rubric. It checks that the converted UUID equals the session ID, the agent reference points to `clinical_note_simple_agent`, and one rubric item is produced. This protects the contract between the data converter and `configs/clinical_note_gym.yaml`.

The tests are small, but they are aimed at the highest-risk local code. The actual GRPO implementation is upstream NeMo RL, so local tests sensibly focus on local schema conversion.

## 28. End-To-End Training Algorithm

An end-to-end training iteration can be understood as a sequence of algorithmic steps.

First, the data loader samples `8` prompts from `data/train.jsonl`. For each prompt, the policy model generates `4` candidate notes using vLLM. This creates `32` candidate completions.

Second, for each candidate note, NeMo Gym evaluates rubric criteria. For each criterion, the judge prompt includes the source conversation, candidate note, criterion question, and expected verdict. The judge emits `[[YES]]` or `[[NO]]`. The resource server aggregates these binary results into a scalar reward.

Third, for each prompt group, rewards are normalized relative to the other completions for the same prompt. A candidate note that is better than its siblings gets positive advantage. A candidate note that is worse than its siblings gets negative advantage.

Fourth, NeMo RL computes token log probabilities under the trainable policy and likely under the reference policy needed for the KL penalty. The loss pushes up the probability of positively advantaged generated tokens and pushes down the probability of negatively advantaged tokens, while the KL term discourages excessive drift from the reference.

Fifth, AdamW updates the language-model parameters. The vision and audio towers remain frozen. Logs, validation, and checkpoints occur according to the configured periods.

In compact mathematical form, each prompt \(x_i\) produces completions \(y_{i,1}, \ldots, y_{i,4}\). The environment produces rewards \(r_{i,1}, \ldots, r_{i,4}\). The group-relative advantage \(A_{i,j}\) is computed from those rewards. The policy update approximately minimizes:

```text
L(theta) =
  - sum over i,j,t of A_{i,j} log pi_theta(y_{i,j,t} | x_i, y_{i,j,<t})
  + beta KL(pi_theta || pi_ref)
```

This expression is simplified, because production GRPO implementations include clipping, masking, old-policy log probabilities, batching details, and distributed synchronization. However, it captures the central idea used by this repository.

## 29. Operational Flow For A User

A normal run begins with credentials. The user creates `.env` from `.env.example`, places a replacement Hugging Face read token in it, and ensures that the account has accepted both the EkaCare dataset and Gemma model access terms.

Next, the user runs `scripts/prepare_data.sh`. This produces `data/train.jsonl`, `data/validation.jsonl`, and `data/manifest.json`. After that, the user runs `scripts/native_preflight.sh`. This validates the environment before expensive training.

For a smoke test, the README suggests running two GRPO iterations through `scripts/train.sh` with overrides such as `grpo.max_num_steps=2`, `grpo.max_num_epochs=1`, and Weights & Biases disabled. Once that succeeds, running `scripts/train.sh` without overrides starts the configured experiment.

The important conceptual point is that configuration overrides are passed through to the upstream NeMo RL launcher. The shell script does not parse GRPO hyperparameters itself. It simply forwards trailing arguments after the config path.

## 30. Failure Modes And Design Responses

The repository is designed around several known failure modes.

A missing Hugging Face token is caught before dataset or model loading. A placeholder token is rejected explicitly. A token from the wrong account produces a tailored gated-repository message in the Python converter.

An old NeMo RL or Transformers environment is caught by checking `AutoConfig.for_model("gemma4")`. This maps directly to the known `KeyError: 'gemma4'` issue.

CUDA driver incompatibility is handled by looking for CUDA 13.2 forward-compatibility libraries and adding them to `LD_LIBRARY_PATH`. If that is not enough, the README explains that an administrator must install the compatibility package or upgrade the driver.

Optional CUDA source-build failures are avoided by bootstrapping dense-only worker environments. This is a practical engineering solution: do not build unused packages required only by unrelated model families or kernels.

Reward collapse is diagnosed by checking whether all rewards in a group are identical. Since GRPO depends on relative differences, identical rewards remove the learning signal. The fix is not purely code-level; it may require better rubric items, judge formatting, generation diversity, or a stronger frozen judge.

## 31. Advanced Perspective: Why This Implementation Is Mostly Glue

A useful way to understand this repository is to separate algorithm ownership from experiment ownership. NeMo RL owns the GRPO implementation, distributed rollout collection, policy updates, and Ray actor machinery. NeMo Gym owns the rubric-server and agent-execution abstractions. vLLM owns efficient generation serving. Hugging Face owns dataset and model distribution. The Ekacare repository owns the integration.

That integration layer is still significant. Machine-learning systems often fail not because the mathematical algorithm is absent, but because schemas do not match, environments are inconsistent, workers use different dependencies, prompts are incomplete, rewards are malformed, or credentials are mishandled. This repository solves those integration problems for a specific clinical-note GRPO experiment.

The converter ensures the training examples speak NeMo Gym's language. The gym config ensures a generated note can be judged criterion by criterion. The GRPO YAML ensures the upstream optimizer, model, generation backend, data paths, and cluster topology agree. The shell scripts ensure the host environment is validated before launching an expensive distributed job. The launcher shim ensures Ray actors use the intended dense worker interpreters.

## 32. What To Study Next

To go deeper, the next implementation layer is the upstream NeMo RL file referenced by `scripts/run_grpo_dense.py`, especially `examples/nemo_gym/run_grpo_nemo_gym.py` and the GRPO algorithm modules in the NeMo RL checkout. That is where rollout collection, old-policy log probabilities, loss computation, KL accounting, checkpointing, and distributed workers are implemented.

Within this repository itself, the most important files to understand in order are `scripts/prepare_dataset.py`, `configs/clinical_note_gym.yaml`, `configs/grpo_gemma4_e4b_8xa100.yaml`, `scripts/bootstrap_dense_envs.sh`, `scripts/run_grpo_dense.py`, and `scripts/train.sh`. Read in that order, the project reveals a clean progression from data schema, to reward semantics, to training configuration, to distributed execution.

The central intuition to keep is this: the model is not being trained from gold clinical notes in a standard supervised way. It is being trained by sampling several candidate notes, judging each one against clinical rubrics, and shifting probability mass toward candidates that score better than their siblings. The local code exists to make that loop concrete, reproducible, and runnable on one eight-A100 machine.
