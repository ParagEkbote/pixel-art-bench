#!/usr/bin/env python3
"""
Export inspect-ai.eval logs to detailed YAML
- reads binary.eval files correctly
- reproduces the console table: json_validity / render_success / pixel_art_quality
- includes per-category breakdown and token counts
"""
import yaml
from pathlib import Path
from datetime import date, datetime
from statistics import mean, pstdev
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["json_validity", "render_success", "pixel_art_quality"]

def find_logs():
    for base in [ROOT / "logs", ROOT / "eval" / "results"]:
        if base.exists():
            yield from base.rglob("*.eval")

def get_val(score):
    if score is None:
        return 0.0
    # inspect Score object, dict, or raw value
    v = getattr(score, "value", None)
    if v is None:
        v = score if not isinstance(score, dict) else score.get("value", 0)

    # map C/I to numbers, then try float
    if isinstance(v, str):
        if v.upper() in ("C", "CORRECT", "PASS", "TRUE", "Y", "1"):
            return 1.0
        if v.upper() in ("I", "INCORRECT", "FAIL", "FALSE", "N", "0"):
            return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0

def summarize(samples, metric):
    vals = [get_val(s.scores.get(metric)) for s in samples]
    m = mean(vals) if vals else 0.0
    se = (pstdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0

    by_cat = {}
    for s in samples:
        cat = (s.metadata or {}).get("category", "unknown")
        by_cat.setdefault(cat, []).append(get_val(s.scores.get(metric)))

    return {
        "accuracy": round(m, 3),
        "mean": round(m, 3),
        "stderr": round(se, 3),
        "by_category": {k: round(mean(v), 3) for k, v in sorted(by_cat.items())}
    }

def parse_time(t):
    if isinstance(t, datetime):
        return t
    if isinstance(t, str):
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    return None

def process_file(fp: Path):
    log = read_eval_log(str(fp))
    full_model = log.eval.model or "unknown"
    short_model = full_model.split("/")[-1]

    usage = next(iter(log.stats.model_usage.values()), None)
    tokens = {
        "input": getattr(usage, "input_tokens", 0),
        "output": getattr(usage, "output_tokens", 0),
    }
    tokens["total"] = tokens["input"] + tokens["output"]

    start = parse_time(log.stats.started_at)
    end = parse_time(log.stats.completed_at)
    total_seconds = int((end - start).total_seconds()) if start and end else 0

    metrics = {m: summarize(log.samples, m) for m in METRICS}

    data = {
        "model": full_model,
        "total_time_seconds": total_seconds,
        "tokens": tokens,
        "metrics": metrics,
        "samples": len(log.samples),
        "date": str(date.today()),
        "source": {
            "url": "https://huggingface.co/AINovice2005/pixel-art-bench",
            "name": "Pixel Art Bench eval traces",
            "user": "AINovice2005"
        }
    }
    return short_model, data

def main():
    logs = list(find_logs())
    print(f"[INFO] found {len(logs)}.eval files")
    aggregated = {}
    for fp in logs:
        try:
            model, data = process_file(fp)
            aggregated.setdefault(model, []).append(data)
            print(f" • {fp.name} → {model}")
        except Exception as e:
            print(f"[SKIP] {fp.name}: {e}")

    for model, runs in aggregated.items():
        out = runs[0].copy()
        out["runs"] = len(runs)
        out_dir = ROOT / "outputs" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pixel-art-bench-detailed.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
        print(f"[OK] {model} → {out_path}")

if __name__ == "__main__":
    main()