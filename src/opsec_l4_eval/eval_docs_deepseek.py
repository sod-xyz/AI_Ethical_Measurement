# src/opsec_l4_eval/eval_docs_deepseek.py

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI   # used with DeepSeek's OpenAI-compatible API
from tqdm import tqdm
from PyPDF2 import PdfReader
from datetime import datetime

# Global DeepSeek client
client: OpenAI | None = None


def read_pdf(path: Path) -> str:
    """Extract plain text from a PDF file."""
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    return "\n\n".join(pages).strip()


def load_doc_judge_template(path: Path) -> str:
    """Load the documentation-judge template text."""
    if not path.exists():
        raise FileNotFoundError(f"Doc judge template not found: {path}")
    return path.read_text(encoding="utf-8")


def call_deepseek_chat(
    model: str,
    system_prompt: str | None,
    user_content: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Call the DeepSeek chat completion API via OpenAI-compatible client.
    """
    global client
    if client is None:
        raise RuntimeError("DeepSeek client not initialized")

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def safe_json_parse(text: str) -> Dict[str, Any]:
    """Try parsing JSON. If fails, return raw text wrapper."""
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting largest { ... } span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return {"raw_output": text}


def evaluate_docs_for_l4(
    l4_root: Path,
    judge_template_path: Path,
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate all document-set folders under `<l4_root>/doc_src`.
    For each subfolder, combine its PDFs and run them through the doc scoring model.
    """
    doc_src_root = l4_root / "doc_src"
    out_dir = l4_root / "doc_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_template = load_doc_judge_template(judge_template_path)

    # Each subfolder in doc_src is treated as one "documentation set" (model vendor)
    doc_sets = [d for d in doc_src_root.iterdir() if d.is_dir()]
    if not doc_sets:
        print(f"No document-set folders found in: {doc_src_root}")
        return {}

    results: List[Dict[str, Any]] = []

    for ds in tqdm(doc_sets, desc="Evaluating document sets"):
        model_id = ds.name
        pdfs = sorted(ds.glob("*.pdf"))

        if not pdfs:
            print(f"[WARN] No PDFs in document set: {model_id}")
            continue

        combined_text_blocks: List[str] = []
        doc_names: List[str] = []

        # Read and concatenate PDFs in this set
        for pdf in pdfs:
            try:
                txt = read_pdf(pdf)
                combined_text_blocks.append(txt)
                doc_names.append(pdf.name)
            except Exception as e:
                combined_text_blocks.append(f"[ERROR READING PDF {pdf.name}] {e}")
                doc_names.append(pdf.name)

        combined_text = "\n\n\n".join(combined_text_blocks)

        # Inject into judge template
        judge_input = judge_template.replace("{document_text}", combined_text)

        try:
            judged = call_deepseek_chat(
                model=model,
                system_prompt=(
                    "You are an L4 documentation evaluator. "
                    "You MUST output valid JSON only."
                ),
                user_content=judge_input,
                temperature=temperature,
            )
        except Exception as e:
            print(f"[ERROR] DeepSeek call failed for set {model_id}: {e}")
            judged = f"__ERROR__: {e}"

        parsed = safe_json_parse(judged)

        entry = {
            "model_id": model_id,
            "model_name": model_id.replace("_", " ").title(),
            "evaluation_target": "Documentation Set",
            "documents": doc_names,
            "score": parsed.get("score"),
            "detailed_assessment": parsed.get("detailed_assessment", {}),
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
            "flags": parsed.get("flags", {}),
        }

        results.append(entry)

    final_json = {
        "metadata": {
            "evaluation_id": f"DOC_L4_{l4_root.name}",
            "indicator_name": l4_root.name.replace("_", " "),
            "evaluation_type": "Documentation Review – Detailed",
            "timestamp": datetime.now().isoformat(),
            "judge_model": model,
            "document_sets_evaluated": [
                {
                    "model_id": r["model_id"],
                    "documents": r["documents"],
                }
                for r in results
            ],
        },
        "results": results,
    }

    out_path = out_dir / "doc_evaluation_results_deepseek.json"
    out_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False))

    print("\n🎉 Documentation evaluation complete (DeepSeek).")
    print(f"📄 Output JSON: {out_path}")

    return final_json


def main() -> None:
    global client

    parser = argparse.ArgumentParser(
        description="Evaluate L4 documentation using DeepSeek."
    )
    parser.add_argument("--l4-root", required=True,
                        help="Path to the L4 folder (e.g., ./L4_SEC_ContinuityRecovery)")
    parser.add_argument("--doc-judge", required=True,
                        help="Path to the doc scoring model template (e.g., L4_SEC_ContinuityRecovery/doc_analysis/doc_scoring_model.txt)")
    parser.add_argument("--model", default="deepseek-chat",
                        help="DeepSeek model to use (default: deepseek-chat)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key",
                        help="Explicit DeepSeek API key (optional, otherwise use DEEPSEEK_API_KEY)")
    args = parser.parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("ERROR: Missing DeepSeek API key. Use --api-key or set DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    # Initialize DeepSeek client via OpenAI-compatible endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    evaluate_docs_for_l4(
        l4_root=Path(args.l4_root).resolve(),
        judge_template_path=Path(args.doc_judge).resolve(),
        model=args.model,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
