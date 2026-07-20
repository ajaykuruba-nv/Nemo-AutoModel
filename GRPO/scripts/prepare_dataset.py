#!/usr/bin/env python3
"""Convert EkaCare clinical-note examples into NeMo Gym rollout JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "ekacare/clinical_note_generation_dataset"
DEFAULT_SYSTEM_PROMPT = (
    "You are a medical scribe. Convert the supplied clinician-patient conversation "
    "into a faithful structured clinical note. Include only facts supported by the "
    "conversation, preserve clinically important negations and uncertainty, and do "
    "not invent diagnoses, medicines, tests, results, or advice. Follow the requested "
    "output format exactly."
)


def _maybe_json(value: Any) -> Any:
    """Decode JSON-valued strings while leaving ordinary text unchanged."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    if text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_rubrics(value: Any) -> list[dict[str, Any]]:
    """Normalize the dataset's serialized rubric into NeMo Gym yes/no items."""
    value = _maybe_json(value)
    if isinstance(value, dict):
        for key in ("rubrics", "rubric", "criteria", "items"):
            if key in value:
                value = value[key]
                break
        else:
            value = list(value.values())

    if isinstance(value, str):
        lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in value.splitlines()
        ]
        value = [line for line in lines if line]
    if not isinstance(value, list):
        value = [value] if value else []

    normalized: list[dict[str, Any]] = []
    for item in value:
        item = _maybe_json(item)
        if isinstance(item, str):
            question, pass_criteria, weight = item.strip(), "YES", 1.0
        elif isinstance(item, dict):
            question = next(
                (
                    str(item[key]).strip()
                    for key in ("question", "rubric", "criterion", "criteria", "description", "text")
                    if item.get(key)
                ),
                "",
            )
            pass_criteria = str(
                item.get("pass_criteria", item.get("expected", item.get("answer", "YES")))
            ).strip()
            weight = float(item.get("weight", 1.0))
        else:
            question, pass_criteria, weight = str(item).strip(), "YES", 1.0

        if question:
            normalized.append(
                {
                    "question": question,
                    "pass_criteria": pass_criteria or "YES",
                    "weight": weight,
                }
            )
    return normalized


def build_prompt(row: dict[str, Any]) -> str:
    sample_prompt = str(row.get("sample_prompt") or "").strip()
    if sample_prompt:
        return sample_prompt
    conversation = str(row.get("text") or "").strip()
    return (
        "Create a structured clinical note from this conversation. Return only the "
        "clinical note.\n\nCLINICAL CONVERSATION:\n" + conversation
    )


def convert_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = build_prompt(row)
    rubrics = normalize_rubrics(row.get("rubrics"))
    if not prompt:
        raise ValueError(f"row {index} has neither sample_prompt nor text")
    if not rubrics:
        raise ValueError(f"row {index} has no usable rubric items")

    session_id = str(row.get("session_id") or f"row-{index:04d}")
    context = str(row.get("text") or prompt).strip()
    return {
        "uuid": session_id,
        "task_id": index,
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "clinical_note_simple_agent",
        },
        "responses_create_params": {
            "input": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        },
        "rubric": rubrics,
        "context": context,
        "metadata": {
            "session_id": session_id,
            "text_md5": str(row.get("text_md5") or ""),
            "source_dataset": DEFAULT_DATASET,
        },
    }


def load_rows(dataset_id: str, split: str, input_jsonl: Path | None) -> list[dict[str, Any]]:
    if input_jsonl is not None:
        with input_jsonl.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip()
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN after accepting the dataset access conditions on Hugging Face."
        )
    if token == "hf_your_token_here":
        raise RuntimeError("HF_TOKEN is still the placeholder from .env.example.")

    from huggingface_hub import HfApi

    try:
        identity = HfApi().whoami(token=token)
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face rejected HF_TOKEN. Create a new read token, put it in "
            ".env, and ensure it belongs to the account that accepted the dataset terms."
        ) from exc

    account = str(identity.get("name") or identity.get("fullname") or "unknown")
    print(f"Authenticated with Hugging Face as: {account}")

    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    try:
        dataset = load_dataset(dataset_id, split=split, token=token)
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            f"Authenticated as '{account}', but that account cannot read gated dataset "
            f"'{dataset_id}'. Open its Hugging Face page while signed in as '{account}', "
            "accept the access/contact-sharing conditions, and retry. If the token is "
            "fine-grained, grant it read access to this gated repository."
        ) from exc
    return [dict(row) for row in dataset]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Optional local raw JSONL; useful for offline validation/tests.",
    )
    args = parser.parse_args()

    rows = load_rows(args.dataset_id, args.split, args.input_jsonl)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if len(rows) < 2:
        raise ValueError("at least two examples are required")
    if not 1 <= args.validation_size < len(rows):
        raise ValueError("validation-size must be between 1 and dataset_size - 1")

    converted = [convert_row(row, i) for i, row in enumerate(rows)]
    random.Random(args.seed).shuffle(converted)
    validation = converted[: args.validation_size]
    train = converted[args.validation_size :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_jsonl(args.output_dir / "train.jsonl", train)
    val_count = write_jsonl(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "dataset_id": args.dataset_id,
        "source_split": args.split,
        "seed": args.seed,
        "train_samples": train_count,
        "validation_samples": val_count,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
