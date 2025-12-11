#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_json(path: Path):
    # Try UTF-8 first
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback for Windows-1252 / cp1252 style characters (e.g., 0x96 dash)
        with path.open("r", encoding="cp1252") as f:
            return json.load(f)

def analyze_prompts(data: dict):
    """
    Expected prompt JSON schema:
    {
      "l4": "L4_SEC_EndpointScrapingEfficacy" (or other),
      "model_under_test": "gpt-4o",
      "judge_model": "gpt-4o",
      "prompt_count": 30,
      "results": [
        {
          "prompt_id": "...",
          "category": "...",
          "subcategory": "...",
          "trials": [
            {
              "evaluation": {
                 "score": 1-5,
                 "flags": {...}
              }
            }
          ]
        },
        ...
      ]
    }
    """
    scores = []
    score_by_category = defaultdict(list)
    flags_counter = Counter()
    total_trials = 0

    for result in data.get("results", []):
        category = result.get("category", "uncategorized")
        for trial in result.get("trials", []):
            ev = trial.get("evaluation", {})
            s = ev.get("score")
            if s is None:
                continue
            scores.append(s)
            score_by_category[category].append(s)
            total_trials += 1

            flags = ev.get("flags", {}) or {}
            for name, value in flags.items():
                if value:
                    flags_counter[name] += 1

    score_dist = Counter(scores)

    prompt_stats = {
        "model_under_test": data.get("model_under_test"),
        "judge_model": data.get("judge_model"),
        "total_prompts": len(data.get("results", [])),
        "total_trials": total_trials,
        "scores": scores,
        "score_dist": score_dist,
        "score_by_category": score_by_category,
        "flags_counter": flags_counter,
    }
    return prompt_stats


def analyze_docs(doc_data: dict):
    """
    Expected doc JSON schema (simplified):
    {
      "metadata": {...},
      "results": [
        {
          "model_id": "deepseek",
          "model_name": "...",
          "score": 0,
          "detailed_assessment": {
              "<subcriterion>": {"score": 0, "evidence": "", "gaps": "..."},
              ...
          },
          "weaknesses": [...],
          "flags": {...},
          "documents": [...]
        },
        ...
      ]
    }
    """
    models = []
    for r in doc_data.get("results", []):
        model_id = r.get("model_id")
        model_name = r.get("model_name", model_id)
        overall_score = r.get("score", 0)

        subcriteria = r.get("detailed_assessment", {}) or {}
        sub_scores = {k: v.get("score", 0) for k, v in subcriteria.items()}
        sub_gaps = {k: v.get("gaps", "") for k, v in subcriteria.items()}

        models.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "overall_score": overall_score,
                "sub_scores": sub_scores,
                "sub_gaps": sub_gaps,
                "flags": r.get("flags", {}) or {},
                "weaknesses": r.get("weaknesses", []),
                "documents": r.get("documents", []),
            }
        )

    return {
        "evaluation_id": doc_data.get("metadata", {}).get("evaluation_id"),
        "indicator_name": doc_data.get("metadata", {}).get("indicator_name"),
        "timestamp": doc_data.get("metadata", {}).get("timestamp"),
        "models": models,
    }


def format_analysis(prompt_stats, doc_stats, l4_name: str):
    scores = prompt_stats["scores"]
    score_dist = prompt_stats["score_dist"]
    score_by_category = prompt_stats["score_by_category"]
    flags_counter = prompt_stats["flags_counter"]
    total_trials = prompt_stats["total_trials"]
    model_under_test = prompt_stats["model_under_test"]
    judge_model = prompt_stats["judge_model"]

    if scores:
        avg_score = mean(scores)
        min_score = min(scores)
        max_score = max(scores)
    else:
        avg_score = min_score = max_score = None

    lines = []

    # Title
    lines.append(f"# {l4_name} analysis")
    lines.append("")
    lines.append(f"**L4 indicator:** `{l4_name}`")
    lines.append(f"**Model under test:** `{model_under_test}`")
    lines.append(f"**Judge model:** `{judge_model}`")
    lines.append("")

    # Prompt-based evaluation
    lines.append("## 1. Prompt-based evaluation for this L4 category")
    lines.append("")
    lines.append(f"- Prompts evaluated: **{prompt_stats['total_prompts']}**")
    lines.append(f"- Trials evaluated: **{total_trials}**")
    if avg_score is not None:
        lines.append(f"- Mean score: **{avg_score:.2f} / 5**")
        lines.append(f"- Range: **{min_score}–{max_score}**")
    lines.append("")

    # Score distribution
    lines.append("### 1.1 Score distribution")
    lines.append("")
    if score_dist:
        lines.append("| Score | Count |")
        lines.append("|-------|-------|")
        for s in sorted(score_dist.keys()):
            lines.append(f"| {s} | {score_dist[s]} |")
        lines.append("")
    else:
        lines.append("No scores available for this prompt evaluation.\n")

    # Per-category view
    lines.append("### 1.2 Scores by category")
    lines.append("")
    if score_by_category:
        lines.append("| Category | N | Mean score |")
        lines.append("|----------|---|------------|")
        for cat, cat_scores in sorted(score_by_category.items()):
            if not cat_scores:
                continue
            lines.append(f"| {cat} | {len(cat_scores)} | {mean(cat_scores):.2f} |")
        lines.append("")
    else:
        lines.append("No per-category scores available.\n")

    # Flag frequencies
    lines.append("### 1.3 Protective behavior flags")
    lines.append("")
    if total_trials > 0 and flags_counter:
        lines.append("| Flag | True count | Percent of trials |")
        lines.append("|------|------------|-------------------|")
        for flag, c in sorted(flags_counter.items()):
            pct = c / total_trials * 100
            lines.append(f"| {flag} | {c} | {pct:.1f}% |")
        lines.append("")
    else:
        lines.append("No flags present in prompt evaluations.\n")

    # Documentation section
    lines.append("## 2. Documentation review for this L4 category")
    lines.append("")
    if doc_stats["models"]:
        lines.append(
            f"Documentation evaluation id: `{doc_stats['evaluation_id']}` "
            f"for indicator: `{doc_stats.get('indicator_name', '')}`, "
            f"timestamp: {doc_stats.get('timestamp', 'n/a')}."
        )
        lines.append("")
        for idx, m in enumerate(doc_stats["models"], start=1):
            lines.append(f"### 2.{idx} {m['model_name']} documentation")
            lines.append("")
            lines.append(f"- Model id: `{m['model_id']}`")
            lines.append(f"- Documents evaluated:")
            for d in m["documents"]:
                lines.append(f"  - {d}")
            lines.append(f"- Overall documentation score: **{m['overall_score']}**")
            lines.append("")
            if m["sub_scores"]:
                lines.append("| Subcriterion | Score | Gap |")
                lines.append("|-------------|-------|-----|")
                for k, s in m["sub_scores"].items():
                    gap = (m["sub_gaps"].get(k) or "").replace("\n", " ")
                    lines.append(f"| {k} | {s} | {gap} |")
                lines.append("")
            if m["weaknesses"]:
                lines.append("Weaknesses noted:")
                for w in m["weaknesses"]:
                    lines.append(f"- {w}")
                lines.append("")
    else:
        lines.append("No documentation results found.\n")

    # High-level summary
    lines.append("## 3. High level interpretation")
    lines.append("")
    if avg_score is not None:
        lines.append(
            f"The prompt-based evaluation for `{l4_name}` produced an "
            f"average behavior score of **{avg_score:.2f} / 5** for `{model_under_test}`."
        )
    else:
        lines.append("No adversarial prompt scores are available for this L4 category.")
    lines.append("")

    if doc_stats["models"]:
        lines.append(
            "The documentation review adds a separate view on how clearly each provider "
            "describes controls related to this L4 indicator. The scores and gaps above "
            "can be used to compare written policy coverage against observed model behavior."
        )
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_l4_generic.py PROMPTS_JSON DOCS_JSON")
        sys.exit(1)

    prompts_path = Path(sys.argv[1])
    docs_path = Path(sys.argv[2])

    prompts_data = load_json(prompts_path)
    docs_data = load_json(docs_path)

    prompt_stats = analyze_prompts(prompts_data)
    doc_stats = analyze_docs(docs_data)

    l4_name = prompts_data.get("l4", "L4_indicator")

    md = format_analysis(prompt_stats, doc_stats, l4_name)
    print(md)


if __name__ == "__main__":
    main()
