# src/opsec_l4_eval/eval_prompts_deepseek.py

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# Global DeepSeek client (initialized in main)
client: OpenAI | None = None


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


def call_deepseek_chat(
    model: str,
    system_prompt: str | None,
    user_content: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    Call DeepSeek via the OpenAI-compatible client.
    Assumes global `client` has been initialized.
    """
    global client
    if client is None:
        raise RuntimeError("DeepSeek client is not initialized")

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
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

    # 1) Direct attempt
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Missing leading brace but starting with "score"
    if raw.startswith('"score"') or raw.startswith("'score'"):
        try:
            return json.loads("{" + raw)
        except Exception:
            pass

    # 3) Extract largest {...} span
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            pass

    # 4) Give up
    return {"raw_output": raw}


def evaluate_prompts_for_l4(
    l4_root: Path,
    prompt_judge_path: Path,
    model_under_test: str,
    judge_model: str,
    temperature_under_test: float = 0.0,
    judge_temperature: float = 0.0,
) -> Dict[str, Any]:
    prompt_data = load_prompt_set(l4_root)
    prompts = prompt_data.get("prompts", [])
    if not prompts:
        print(f"No prompts found under {l4_root}")
        return {
            "l4": l4_root.name,
            "model_under_test": model_under_test,
            "judge_model": judge_model,
            "prompt_count": 0,
            "results": [],
        }

    # DeepSeek results folder (separate from OpenAI)
    out_root = l4_root / "prompt_scoring" / "deepseek_results"
    out_root.mkdir(parents=True, exist_ok=True)

    judge_template = load_prompt_judge_template(prompt_judge_path)

    print(f"Evaluating prompts for: {l4_root.name}")
    print(f"  Model under test: {model_under_test}")
    print(f"  Judge model:      {judge_model}")
    print(f"  Prompt judge:     {prompt_judge_path}")
    print(f"  Prompt count:     {len(prompts)}")
    print()

    combined_results: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    all_flag_keys: Set[str] = set()

    for p in tqdm(prompts, desc=l4_root.name):
        pid = p.get("id")
        prompt_text = p.get("prompt", "")
        category = p.get("category", "")
        subcategory = p.get("subcategory", "")

        # -------- Model under test response --------
        try:
            model_response = call_deepseek_chat(
                model=model_under_test,
                system_prompt="You are a general-purpose AI assistant.",
                user_content=prompt_text,
                temperature=temperature_under_test,
            )
        except Exception as e:
            print(f"[ERROR] Model under test failed for {pid}: {e}")
            model_response = f"__MODEL_ERROR__: {e}"

        # -------- Judge model scoring --------
        judge_input = (
            judge_template
            .replace("{test_prompt}", prompt_text)
            .replace("{model_response}", model_response)
        )

        try:
            judge_raw = call_deepseek_chat(
                model=judge_model,
                system_prompt="You MUST output valid JSON only.",
                user_content=judge_input,
                temperature=judge_temperature,
            )
        except Exception as e:
            print(f"[ERROR] Judge model failed for {pid}: {e}")
            judge_raw = f"__JUDGE_ERROR__: {e}"

        evaluation = parse_model_json(judge_raw)
        flags = evaluation.get("flags", {}) if isinstance(evaluation, dict) else {}
        score = evaluation.get("score") if isinstance(evaluation, dict) else None

        if isinstance(flags, dict):
            all_flag_keys.update(flags.keys())

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

        csv_rows.append(
            {
                "prompt_id": pid,
                "category": category,
                "subcategory": subcategory,
                "prompt": prompt_text,
                "model_response": model_response,
                "avg_score": score,
                "_flags": flags,
            }
        )

    json_out = {
        "l4": l4_root.name,
        "model_under_test": model_under_test,
        "judge_model": judge_model,
        "prompt_count": len(prompts),
        "results": combined_results,
    }

    # -------- Write combined JSON --------
    json_path = out_root / "prompt_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # -------- Write CSV --------
    flag_cols = sorted(all_flag_keys)
    fieldnames = [
        "prompt_id",
        "category",
        "subcategory",
        "prompt",
        "model_response",
        "avg_score",
    ] + flag_cols

    csv_path = out_root / "prompt_evaluation_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in csv_rows:
            out_row: Dict[str, Any] = {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "prompt": row["prompt"],
                "model_response": row["model_response"],
                "avg_score": row["avg_score"],
            }
            flags = row.get("_flags", {}) or {}
            for fk in flag_cols:
                out_row[fk] = bool(flags.get(fk, False))
            writer.writerow(out_row)

    print("\nPrompt evaluation complete (DeepSeek).")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    return json_out


def main() -> None:
    global client

    parser = argparse.ArgumentParser(
        description="Evaluate prompt behavior for a given L4 using DeepSeek."
    )
    parser.add_argument("--l4-root", required=True,
                        help="Path to the L4 folder (e.g., ./L4_SEC_SecretsManagement).")
    parser.add_argument("--prompt-judge", required=True,
                        help="Path to the prompt-scoring template .txt.")
    parser.add_argument(
        "--model-under-test",
        default="deepseek-chat",
        help="DeepSeek model whose behavior is being tested (default: deepseek-chat).",
    )
    parser.add_argument(
        "--judge-model",
        default="deepseek-chat",
        help="DeepSeek model that scores the responses (default: deepseek-chat).",
    )
    parser.add_argument(
        "--temperature-under-test",
        type=float,
        default=0.0,
        help="Temperature for the model under test.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the judge model.",
    )
    parser.add_argument(
        "--api-key",
        help="DeepSeek API key (otherwise uses DEEPSEEK_API_KEY env var via .env).",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: No DeepSeek API key found. Use --api-key or set DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    # Initialize DeepSeek client (OpenAI-compatible)
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

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
