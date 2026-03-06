"""
eval_ocr_all.py
===============
Evaluate all 3 OCR engines (Tesseract, PaddleOCR, docTR) on the full FUNSD
dataset (199 images = 149 training + 50 testing) and produce a side-by-side
comparison report with actionable model-selection inferences.

Usage
-----
    python eval_ocr_all.py                            # uses default FUNSD path
    python eval_ocr_all.py --data_dir /path/to/FUNSD
    python eval_ocr_all.py --limit 5                  # quick smoke-test
    python eval_ocr_all.py --engines tesseract doctr  # subset of engines
    python eval_ocr_all.py --output_dir ./my_results

Output
------
    <output_dir>/ocr_eval_tesseract.json
    <output_dir>/ocr_eval_paddle.json
    <output_dir>/ocr_eval_doctr.json
    <output_dir>/ocr_comparison_all.json
"""

import os
import sys
import json
import time
import argparse

# ---------------------------------------------------------------------------
# Add parent / sibling paths so ocr_evaluation.py is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ocr_evaluation import (
    OCREvaluator,
    load_funsd_documents,
    print_summary,
    print_comparison,
    save_report,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# Prefer an explicit FUNSD root if provided, else use a repo-local default.
DEFAULT_FUNSD_ROOT = os.getenv("FUNSD_ROOT") or os.path.join(REPO_ROOT, "data", "FUNSD")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eval_results")

ALL_ENGINES = ["tesseract", "paddle", "paddle-v4", "paddle-vl", "doctr"]

METRIC_NAMES = {
    "cer":                  "CER ↓",
    "wer":                  "WER ↓",
    "word_recall":          "Word Recall ↑",
    "over_segmentation":    "Over-seg ↓",
    "under_segmentation":   "Under-seg ↓",
    "order_consistency":    "Order Consistency ↑",
    "mean_iou":             "Mean IoU ↑",
    "median_iou":           "Median IoU ↑",
    "pct_iou_above_0.7":    "% IoU > 0.7 ↑",
}

# higher-is-better metrics; all others are lower-is-better
HIGHER_IS_BETTER = {"word_recall", "order_consistency", "mean_iou", "median_iou", "pct_iou_above_0.7"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_splits(data_dir: str, limit=None):
    """Load training_data + testing_data (199 images total)."""
    docs = []
    for split in ("training_data", "testing_data"):
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            print(f"  [WARN] Split directory not found: {split_path}")
            continue
        split_docs = load_funsd_documents(data_dir, split=split)
        print(f"  Loaded {len(split_docs):3d} documents from '{split}'")
        docs.extend(split_docs)
    if limit and limit > 0:
        docs = docs[:limit]
        print(f"  ⚠  Limiting to first {len(docs)} documents (--limit).\n")
    print(f"  Total: {len(docs)} documents.\n")
    return docs


def _save_combined(results: dict, output_dir: str):
    """Save all engines' overall stats into a single comparison JSON."""
    combined = {
        engine: data["overall"]
        for engine, data in results.items()
    }
    path = os.path.join(output_dir, "ocr_comparison_all.json")
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined comparison saved → {path}")


def _print_rich_comparison(results: dict):
    """
    Print an annotated comparison table.
    Best value per row is highlighted with ✓.
    """
    engines = list(results.keys())
    overall = {e: results[e]["overall"] for e in engines}

    col_width = 14
    header = f"{'Metric':<26}"
    for eng in engines:
        header += f"  {eng:>{col_width}}"
    sep = "=" * (26 + (col_width + 2) * len(engines))

    print()
    print(sep)
    print("  OCR ENGINE COMPARISON — FUNSD  ")
    print(sep)
    print(header)
    print("-" * len(sep))

    for key, label in METRIC_NAMES.items():
        row = f"{label:<26}"
        vals = {e: overall[e].get(key, 0.0) for e in engines}
        if key in HIGHER_IS_BETTER:
            best_engine = max(vals, key=vals.__getitem__)
        else:
            best_engine = min(vals, key=vals.__getitem__)

        for eng in engines:
            v = vals[eng]
            marker = " ✓" if eng == best_engine else "  "
            row += f"  {v:>{col_width - 2}.4f}{marker}"
        print(row)

    print(sep)

    # Word count stats
    print()
    print(f"{'Documents Evaluated':<26}" + "".join(
        f"  {overall[e]['total_documents']:>{col_width}}" for e in engines))
    print(f"{'GT Words (total)':<26}" + "".join(
        f"  {overall[e]['total_gt_words']:>{col_width}}" for e in engines))
    print(f"{'OCR Words (total)':<26}" + "".join(
        f"  {overall[e]['total_ocr_words']:>{col_width}}" for e in engines))
    print(f"{'OCR / GT ratio':<26}" + "".join(
        f"  {overall[e]['ocr_to_gt_ratio']:>{col_width}.4f}" for e in engines))
    print(f"{'Matched Words':<26}" + "".join(
        f"  {overall[e]['total_matched_words']:>{col_width}}" for e in engines))
    print(f"{'Missed Words':<26}" + "".join(
        f"  {overall[e]['total_missed_words']:>{col_width}}" for e in engines))
    print(sep)


def _print_inferences(results: dict):
    """Print actionable conclusions derived from the evaluation numbers."""
    engines = list(results.keys())
    overall = {e: results[e]["overall"] for e in engines}

    # Rank by CER (primary), then word_recall (secondary)
    ranked_cer    = sorted(engines, key=lambda e: overall[e]["cer"])
    ranked_recall = sorted(engines, key=lambda e: overall[e]["word_recall"], reverse=True)
    ranked_iou    = sorted(engines, key=lambda e: overall[e]["mean_iou"], reverse=True)

    best_cer    = ranked_cer[0]
    best_recall = ranked_recall[0]
    best_iou    = ranked_iou[0]

    print()
    print("=" * 60)
    print("  🔍  INFERENCES & RECOMMENDATIONS")
    print("=" * 60)

    print(f"""
1. TEXT ACCURACY (CER / WER)
   Best engine : {best_cer.upper()} (CER={overall[best_cer]['cer']:.4f})
   • Lower CER means fewer character-level OCR errors — critical
     when OCR text is directly fed to downstream NLP/LLM models.
   • Engines sorted: {" > ".join(e.upper() for e in ranked_cer)}
     (lower is better)

2. WORD RECALL
   Best engine : {best_recall.upper()} (Recall={overall[best_recall]['word_recall']:.4f})
   • Higher recall means fewer GT words are missed by the OCR.
     Missing words = silent failures in LayoutLM token alignment.
   • Engines sorted: {" > ".join(e.upper() for e in ranked_recall)}
     (higher is better)

3. SPATIAL ACCURACY (IoU)
   Best engine : {best_iou.upper()} (Mean IoU={overall[best_iou]['mean_iou']:.4f})
   • IoU measures how well OCR bounding boxes match GT boxes.
     LayoutLMv3 uses bboxes as a primary position signal — poor
     spatial accuracy degrades the model even when text is correct.
   • Engines sorted: {" > ".join(e.upper() for e in ranked_iou)}
     (higher is better)

4. SEGMENTATION
   • Over-segmentation (word splits): hurts token alignment — one
     GT entity becomes multiple OCR tokens, diluting label signal.
   • Under-segmentation (word merges): causes missed entity starts.
""")

    # Overall recommendation
    # Simple scoring: rank each engine across 3 key metrics; lowest total rank = best
    scores = {e: 0 for e in engines}
    for rank, e in enumerate(ranked_cer):
        scores[e] += rank
    for rank, e in enumerate(ranked_recall[::-1]):   # reversed: higher recall = lower rank
        scores[e] += rank
    for rank, e in enumerate(ranked_iou[::-1]):
        scores[e] += rank
    recommended = min(scores, key=scores.__getitem__)

    print(f"  ★  OVERALL RECOMMENDATION")
    print(f"     → Use '{recommended.upper()}' as the OCR for LayoutLM fine-tuning.")
    print(f"       (Best combined rank across CER, Word Recall, and IoU)")

    # Specific LayoutLM tip per engine
    per_model_map = {
        "tesseract":  "layoutlmv3-funsd-tesseract",
        "paddle":     "layoutlmv3-funsd-paddle  (PP-OCRv3)",
        "paddle-v4":  "layoutlmv3-funsd-paddle  (retrain with PP-OCRv4 for best results)",
        "paddle-vl":  "layoutlmv3-funsd-paddle  (retrain with PaddleOCR VL once on 3.x)",
        "doctr":      "layoutlmv3-funsd-doctr (or -doctr-large for higher accuracy)",
    }
    print(f"\n  Corresponding LayoutLM models per OCR engine:")
    for eng in engines:
        marker = " ← recommended" if eng == recommended else ""
        print(f"    {eng.upper():12s}  →  {per_model_map.get(eng, eng)}{marker}")

    print()
    print("  NOTE: The layoutlmv3-funsd model (GT bboxes, F1=0.82) is the")
    print("  ceiling — it was trained with perfect bounding boxes. Any OCR-")
    print("  trained model will score lower but can run on unseen documents")
    print("  without ground-truth annotations.")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all OCR engines on the full FUNSD dataset (199 images)."
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_FUNSD_ROOT,
        help=f"Path to FUNSD root directory (default: {DEFAULT_FUNSD_ROOT})"
    )
    parser.add_argument(
        "--engines", nargs="+", default=ALL_ENGINES,
        choices=["tesseract", "paddle", "paddle-v4", "paddle-vl", "doctr"],
        help="OCR engines to evaluate (default: all three)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of documents per split (for quick smoke tests)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save JSON reports (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  OCR EVALUATION — All Engines vs FUNSD (199 images)")
    print("=" * 60)
    print(f"  FUNSD dir  : {args.data_dir}")
    print(f"  Engines    : {args.engines}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Limit      : {'none' if args.limit is None else args.limit}")
    print()

    # Load all 199 documents once and share across engines
    print("[ Loading FUNSD documents ]")
    documents = _load_all_splits(args.data_dir, limit=args.limit)
    if not documents:
        print("ERROR: No documents found. Check --data_dir path.")
        sys.exit(1)

    total_start = time.time()
    all_results = {}

    for engine in args.engines:
        print(f"\n{'=' * 60}")
        print(f"  Running: {engine.upper()}")
        print(f"{'=' * 60}")

        t0 = time.time()
        evaluator = OCREvaluator(ocr_engine=engine)
        report = evaluator.evaluate_dataset(documents)
        elapsed = time.time() - t0

        all_results[engine] = report

        print_summary(engine, report["overall"])
        save_report(engine, report, args.output_dir)

        # Rename saved file to match our naming convention
        old_path = os.path.join(args.output_dir, f"ocr_eval_report_{engine}.json")
        new_path = os.path.join(args.output_dir, f"ocr_eval_{engine}.json")
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"  Renamed → {new_path}")

        print(f"  ⏱  Elapsed: {elapsed:.1f}s  ({elapsed / max(len(documents), 1):.2f}s/doc)")

    # -----------------------------------------------------------------------
    # Comparison table + inferences
    # -----------------------------------------------------------------------
    _print_rich_comparison(all_results)
    _print_inferences(all_results)
    _save_combined(all_results, args.output_dir)

    total_elapsed = time.time() - total_start
    print(f"\n⏱  Total wall-clock time: {total_elapsed:.1f}s")
    print(f"   Engines evaluated: {len(all_results)}")
    print(f"   Documents per engine: {len(documents)}")
    print()


if __name__ == "__main__":
    main()
