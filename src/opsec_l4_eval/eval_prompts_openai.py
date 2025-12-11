# src/opsec_l4_eval/eval_prompts_openai.py

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv
from tqdm import tqdm


def load_prompt_set(l4_root: Path) -> Dict[str, Any]:
    prompt_path = l4_root / "prompt_src" / "prompt_set.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt_set.json not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_judge_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt judge template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_openai_chat(model: str, system_prompt: str | None, user_content: str,
                     temperature: float = 0.0, max_tokens: int = 2048) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def parse_model_json(raw: str) -> Dict[str, Any]:
    """Extract valid JSON from messy LLM output."""
    if raw is None:
        return {"raw_output": ""}

    raw = raw.strip()

    # 1 direct load
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2 missing opening brace
    if raw.startswith('"score"') or raw.startswith("'score'"):
        try:
            return json.loads("{" + raw)
        except Exception:
            pass

    # 3 extract { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass

    # 4 give up
    return {"raw_output": raw}


def evaluate_prompts_for_l4(
    l4_root: Path,
    prompt_judge_path: Path,
    model_under_test: str,
    judge_model: str,
    temperature_under_test: float = 0.0,
    judge_temperature: float = 0.0,
):
    prompt_data = load_prompt_set(l4_root)
    prompts = prompt_data.get("prompts", [])
    if not prompts:
        print(f"No prompts found under {l4_root}")
        return

    out_root = l4_root / "prompt_scoring" / "openai_results"
    out_root.mkdir(parents=True, exist_ok=True)

    judge_template = load_prompt_judge_template(prompt_judge_path)

    print(f"Evaluating prompts for: {l4_root.name}")
    print(f"Model under test: {model_under_test}")
    print(f"Judge model: {judge_model}")
    print(f"Prompt count: {len(prompts)}")
    print()

    combined_results = []
    csv_rows = []
    all_flag_keys = set()

    for p in tqdm(prompts, desc=l4_root.name):
        pid = p["id"]
        prompt_text = p["prompt"]
        category = p.get("category", "")
        subcategory = p.get("subcategory", "")

        # --- model response ---
        try:
            model_response = call_openai_chat(
                model=model_under_test,
                system_prompt="You are a general-purpose AI assistant.",
                user_content=prompt_text,
                temperature=temperature_under_test,
            )
        except Exception as e:
            model_response = f"__MODEL_ERROR__: {e}"

        # --- judge ---
        judge_input = (
            judge_template
            .replace("{test_prompt}", prompt_text)
            .replace("{model_response}", model_response)
        )

        try:
            judge_raw = call_openai_chat(
                model=judge_model,
                system_prompt="You MUST output valid JSON only.",
                user_content=judge_input,
                temperature=judge_temperature,
            )
        except Exception as e:
            judge_raw = f"__JUDGE_ERROR__: {e}"

        evaluation = parse_model_json(judge_raw)
        flags = evaluation.get("flags", {}) if isinstance(evaluation, dict) else {}
        score = evaluation.get("score") if isinstance(evaluation, dict) else None

        # Track flags so CSV has dynamic columns
        if isinstance(flags, dict):
            all_flag_keys.update(flags.keys())

        # JSON object entry
        trial = {
            "prompt_id": pid,
            "category": category,
            "subcategory": subcategory,
            "test_model": model_under_test,
            "judge_model": judge_model,
            "test_prompt": prompt_text,
            "model_response": model_response,
            "evaluation": evaluation,
        }

        combined_results.append(
            {
                "prompt_id": pid,
                "category": category,
                "subcategory": subcategory,
                "trials": [trial],
            }
        )

        csv_rows.append({
            "prompt_id": pid,
            "category": category,
            "subcategory": subcategory,
            "prompt": prompt_text,
            "model_response": model_response,
            "avg_score": score,
            "_flags": flags,
        })

    # -------- WRITE JSON --------
    json_out = {
        "l4": l4_root.name,
        "model_under_test": model_under_test,
        "judge_model": judge_model,
        "prompt_count": len(prompts),
        "results": combined_results,
    }

    json_path = out_root / "prompt_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # -------- WRITE CSV --------
    flag_cols = sorted(all_flag_keys)

    csv_path = out_root / "prompt_evaluation_results.csv"
    fieldnames = [
        "prompt_id",
        "category",
        "subcategory",
        "prompt",
        "model_response",
        "avg_score",
    ] + flag_cols

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in csv_rows:
            out = {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "prompt": row["prompt"],
                "model_response": row["model_response"],
                "avg_score": row["avg_score"],
            }
            flags = row["_flags"]
            for fk in flag_cols:
                out[fk] = bool(flags.get(fk, False))
            writer.writerow(out)

    print("\nDone.")
    print("JSON:", json_path)
    print("CSV:", csv_path)
    return json_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l4-root", required=True)
    parser.add_argument("--prompt-judge", required=True)
    parser.add_argument("--model-under-test", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--temperature-under-test", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--api-key")

    args = parser.parse_args()

    load_dotenv()
    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        print("ERROR: No API key found.")
        sys.exit(1)

    evaluate_prompts_for_l4(
        l4_root=Path(args.l4_root).resolve(),
        prompt_judge_path=Path(args.prompt_judge).resolve(),
        model_under_test=args.model_under_test,
        judge_model=args.judge_model,
        temperature_under_test=args.temperature_under_test,
        judge_temperature=args.judge_temperature,
    )


if __name__ == "__main__":
    main()
