"""
eval_layoutlm_all.py
====================
Evaluate all LayoutLMv3 model × OCR combinations on the FUNSD test split.

Test Matrix (8 runs)
---------------------
Native-OCR pairs (model trained with its own OCR):
  1. layoutlmv3-funsd-doctr       + docTR
  2. layoutlmv3-funsd-doctr-large + docTR
  3. layoutlmv3-funsd-paddle      + PaddleOCR
  4. layoutlmv3-funsd-tesseract   + Tesseract

layoutlmv3-funsd (trained on GT bboxes) × all OCRs:
  5. layoutlmv3-funsd + Tesseract   (cross-OCR stress test)
  6. layoutlmv3-funsd + PaddleOCR
  7. layoutlmv3-funsd + docTR
  8. layoutlmv3-funsd + GT bboxes   (true baseline)

Usage
-----
    # Full run (all 8 pairs, FUNSD test split)
    python eval_layoutlm_all.py \\
        --cache_dir ./.hf_cache

    # Quick smoke test (3 docs, 1 pair)
    python eval_layoutlm_all.py \\
        --cache_dir /path/to/cache \\
        --limit 3 \\
        --pairs "funsd-tesseract:tesseract"

    # Custom output
    python eval_layoutlm_all.py \\
        --cache_dir /path/to/cache \\
        --output_dir ./my_eval

Output
------
    eval_results/layoutlm_<model>_<ocr>.json   — per-pair detail
    eval_results/layoutlm_comparison.json      — combined matrix
"""

import os
import sys
import json
import time
import glob
import argparse
import statistics
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)
# Optional DocFormer imports
try:
    from transformers import DocFormerProcessor, DocFormerForTokenClassification
except ImportError:
    DocFormerProcessor = None
    DocFormerForTokenClassification = None
import evaluate as hf_evaluate

from layoutlm_customOCR import (
    OCRBackend,
    align_ocr_to_ground_truth,
)

# JSON encoder that handles numpy int64 / float32 from seqeval
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

L_DIR = SCRIPT_DIR   # LayoutLM directory

DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eval_results")
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Prefer an explicit cache dir if provided, else use a repo-local cache.
DEFAULT_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, ".hf_cache")

# (model_dir_name, ocr_engine_or_"gt")
# "gt" = pass ground-truth words + bboxes directly (no OCR)
ALL_PAIRS = [
    # ---- GT-bbox model vs all OCRs ----
    ("layoutlmv3-funsd",               "tesseract"),
    ("layoutlmv3-funsd",               "paddle"),
    ("layoutlmv3-funsd",               "doctr"),
    # ---- True baseline: GT-bbox model with GT bboxes ----
    ("layoutlmv3-funsd",               "gt"),
]

METRIC_KEYS = ["precision", "recall", "f1", "accuracy"]

# Higher is better for all of these
HIGHER_IS_BETTER = {"precision", "recall", "f1", "accuracy", "word_recall", "mean_iou"}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(model_dir: str) -> str:
    """
    Scan all trainer_state.json files under model_dir and return the
    checkpoint subdirectory with the highest best_metric (eval F1).
    Falls back to the numerically highest checkpoint if no state is found.
    """
    best_f1   = -1.0
    best_ckpt = None

    for state_path in glob.glob(os.path.join(model_dir, "checkpoint-*", "trainer_state.json")):
        try:
            with open(state_path) as f:
                state = json.load(f)
            f1 = state.get("best_metric", -1.0) or -1.0
            if f1 > best_f1:
                best_f1   = f1
                best_ckpt = os.path.dirname(state_path)
        except Exception:
            continue

    if best_ckpt is None:
        # Fallback: numerically latest checkpoint
        all_ckpts = glob.glob(os.path.join(model_dir, "checkpoint-*"))
        if all_ckpts:
            best_ckpt = max(all_ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))

    return best_ckpt


def _arch_from_model_dir(model_dir: str) -> str:
    return "docformer" if "docformer" in model_dir else "layoutlmv3"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _get_ocr_tokens(example, ocr_backend: OCRBackend):
    """Run OCR on a FUNSD example image; return (words, boxes_1000, boxes_px)."""
    image = example["image"].convert("RGB")
    words, boxes_1000, _, boxes_px = ocr_backend.run(image)
    return image, words, boxes_1000, boxes_px


def _get_gt_tokens(example):
    """Return ground-truth words and bboxes directly from the FUNSD example."""
    image   = example["image"].convert("RGB")
    words   = example["words"]
    boxes   = example["bboxes"]          # already 0-1000 normalised in FUNSD HF dataset
    return image, words, boxes, boxes    # boxes_px ≈ boxes (0-1000 scale; pixel info not needed)


# ---------------------------------------------------------------------------
# Seqeval metric computation
# ---------------------------------------------------------------------------

_seqeval = hf_evaluate.load("seqeval")


def _compute_seqeval(true_labels_list, pred_labels_list):
    """Wrapper around seqeval returning overall + per-class results."""
    results = _seqeval.compute(predictions=pred_labels_list, references=true_labels_list)
    overall = {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"],    4),
        "f1":        round(results["overall_f1"],        4),
        "accuracy":  round(results["overall_accuracy"],  4),
    }
    per_class = {}
    for key, val in results.items():
        if isinstance(val, dict):
            per_class[key] = {
                "precision": round(val["precision"], 4),
                "recall":    round(val["recall"],    4),
                "f1":        round(val["f1"],        4),
                "number":    val["number"],
            }
    return overall, per_class


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    model_dir: str,
    ocr_engine: str,
    dataset,
    cache_dir: str,
    limit: int = None,
    device=None,
) -> dict:
    """
    Evaluate one (model, OCR) pair on the FUNSD test split.

    Returns a dict with:
        checkpoint, ocr_engine, overall (seqeval), per_class,
        timing, total_docs, total_ocr_words, total_gt_words
    """
    if "layoutgraph" in model_dir:
        try:
            from layout_graph_pipeline import evaluate_layoutgraph_pair
        except Exception as e:
            raise RuntimeError(
                "Failed to import layout_graph_pipeline for layoutgraph evaluation. "
                "Ensure torch-geometric and layout_graph_pipeline dependencies are installed."
            ) from e

        return evaluate_layoutgraph_pair(
            model_dir=model_dir,
            ocr_engine=ocr_engine,
            dataset=dataset,
            cache_dir=cache_dir,
            limit=limit,
            device=device,
        )

    if device is None:
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    abs_model_dir = os.path.join(L_DIR, model_dir)
    checkpoint    = find_best_checkpoint(abs_model_dir)

    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found under {abs_model_dir}. "
            "Did training complete?"
        )

    print(f"  Checkpoint : {os.path.relpath(checkpoint, L_DIR)}")
    arch = _arch_from_model_dir(model_dir)

    # --- Load model ---
    # The processor base is always layoutlmv3-base (apply_ocr=False).
    # For the -large models the base pretrained weights differ, but as a
    # fine-tuned checkpoint the tokenizer/processor config is compatible.
    is_large = "large" in model_dir
    if arch == "docformer":
        if DocFormerProcessor is None or DocFormerForTokenClassification is None:
            raise ImportError("DocFormer not available in this transformers version. Install transformers>=4.37 to evaluate DocFormer runs.")
        base_model_id = "microsoft/docformer-base"
        processor = DocFormerProcessor.from_pretrained(base_model_id, cache_dir=cache_dir)
        model = DocFormerForTokenClassification.from_pretrained(checkpoint)
    else:
        base_model_id = "microsoft/layoutlmv3-large" if is_large else "microsoft/layoutlmv3-base"
        processor = LayoutLMv3Processor.from_pretrained(
            base_model_id, apply_ocr=False, cache_dir=cache_dir
        )
        model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    # --- Auto-detect trained image resolution from pos_embed shape ---
    # Standard: 224px → 14×14 patches + 1 CLS = 197
    # 384px trained → 24×24 + 1 = 577
    # 512px trained → 32×32 + 1 = 1025
    if arch == "docformer":
        img_size = 512
        processor.image_processor.size = {"height": img_size, "width": img_size}
        processor.image_processor.do_resize = True
        processor.image_processor.do_pad = True
    else:
        img_size = 224  # safe default
        if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "pos_embed"):
            n_tokens = model.layoutlmv3.pos_embed.shape[1]   # e.g. 577
            n_patches = n_tokens - 1                          # subtract CLS token
            patch_grid = int(round(n_patches ** 0.5))         # e.g. 24
            if patch_grid * patch_grid == n_patches:
                img_size = patch_grid * 16                    # patch_size=16 always
        if img_size != 224:
            print(f"  Detected trained image size: {img_size}×{img_size}")
        processor.image_processor.size = {"height": img_size, "width": img_size}
        processor.image_processor.do_resize = True
        processor.image_processor.do_pad = True

    # --- Aspect-ratio-preserving resize+pad helper (must match training) ---
    def _pad_to_square(img: Image.Image, target: int = img_size) -> Image.Image:
        """Scale long side to `target`, pad short side with white."""
        w, h = img.size
        scale = target / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        canvas.paste(img.resize((nw, nh), Image.LANCZOS),
                     ((target - nw) // 2, (target - nh) // 2))
        return canvas

    # Re-initialise visual_bbox grid to match trained resolution
    if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "init_visual_bbox"):
        try:
            grid = img_size // 16
            model.layoutlmv3.init_visual_bbox(image_size=(grid, grid))
        except Exception:
            pass   # not all checkpoint versions expose this method


    id2label   = model.config.id2label
    label_list = [id2label[i] for i in range(len(id2label))]
    label2id   = {v: k for k, v in id2label.items()}
    o_label_id = label2id.get("O", 0)

    # --- Optionally initialise OCR backend ---
    ocr_backend = None
    if ocr_engine != "gt":
        print(f"  Initialising OCR backend: {ocr_engine} …")
        ocr_backend = OCRBackend(engine=ocr_engine)

    test_split   = dataset["test"]
    docs_to_eval = test_split if limit is None else test_split.select(range(min(limit, len(test_split))))

    true_labels_all = []
    pred_labels_all = []
    total_ocr_words = 0
    total_gt_words  = 0
    t0 = time.time()

    for idx, example in enumerate(docs_to_eval):
        print(f"    [{idx+1}/{len(docs_to_eval)}] …", end="\r")

        gt_words  = example["words"]
        gt_boxes  = example["bboxes"]  # 0-1000 scale
        gt_labels = example["ner_tags"]
        total_gt_words += len(gt_words)

        # --- Prepare input tokens ---
        if ocr_engine == "gt":
            image      = example["image"].convert("RGB")
            words      = gt_words
            boxes_1000 = gt_boxes
            labels     = gt_labels
        else:
            image, words, boxes_1000, _ = _get_ocr_tokens(example, ocr_backend)
            total_ocr_words += len(words)
            if not words:
                words      = [""]
                boxes_1000 = [[0, 0, 0, 0]]
            labels = align_ocr_to_ground_truth(
                words, boxes_1000,
                gt_words, gt_boxes, gt_labels,
                default_label_id=o_label_id
            )

        # --- Encode ---
        # Use the aspect-ratio-preserved image so visual patches match training.
        image_sq = _pad_to_square(image)
        encoding = processor(
            image_sq, words,
            boxes=boxes_1000,
            word_labels=labels,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        input_label_ids = encoding["labels"].squeeze().tolist()
        inputs = {k: v.to(device) for k, v in encoding.items() if k != "labels"}

        # --- Inference ---
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(-1).squeeze().cpu().tolist()

        # --- Collapse to word-level (first sub-token only) ---
        prev_word_id  = None
        doc_true, doc_pred = [], []
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_word_id:
                true_lid = input_label_ids[pos]
                pred_lid = preds[pos]
                if true_lid != -100:
                    doc_true.append(label_list[true_lid])
                    doc_pred.append(label_list[pred_lid])
                prev_word_id = wid

        true_labels_all.append(doc_true)
        pred_labels_all.append(doc_pred)

    elapsed = time.time() - t0
    print()   # clear \r line

    overall, per_class = _compute_seqeval(true_labels_all, pred_labels_all)

    return {
        "model_dir":        model_dir,
        "ocr_engine":       ocr_engine,
        "checkpoint":       os.path.relpath(checkpoint, L_DIR),
        "overall":          overall,
        "per_class":        per_class,
        "total_docs":       len(docs_to_eval),
        "total_gt_words":   total_gt_words,
        "total_ocr_words":  total_ocr_words if ocr_engine != "gt" else total_gt_words,
        "elapsed_seconds":  round(elapsed, 2),
        "seconds_per_doc":  round(elapsed / max(len(docs_to_eval), 1), 3),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pair_label(model_dir: str, ocr_engine: str) -> str:
    short = model_dir.replace("layoutlmv3-", "")
    return f"{short} + {ocr_engine}"


def _print_comparison_table(results: list):
    """Print ranked comparison table (sorted by F1 descending)."""
    sorted_results = sorted(results, key=lambda r: r["overall"]["f1"], reverse=True)

    label_w = 40
    col_w   = 10

    cols = ["F1", "Prec", "Recall", "Acc"]
    keys = ["f1", "precision", "recall", "accuracy"]

    header_label = "Model + OCR"
    header = f"  {'#':<3}  {header_label:<{label_w}}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = "=" * len(header)

    print()
    print(sep)
    print("  LAYOUTLM EVALUATION — RANKED BY F1")
    print(sep)
    print(header)
    print("-" * len(sep))

    for rank, r in enumerate(sorted_results, start=1):
        lbl = _pair_label(r["model_dir"], r["ocr_engine"])
        row = f"  {rank:<3}  {lbl:<{label_w}}"
        for key in keys:
            val = r["overall"].get(key, 0.0)
            marker = " ★" if rank == 1 and key == "f1" else "  "
            row += f"  {val:>{col_w - 2}.4f}{marker}"
        print(row)

    print(sep)


def _print_per_class_table(results: list):
    """Print per-class F1 for all pairs."""
    # Collect all class labels
    all_classes = set()
    for r in results:
        all_classes.update(r["per_class"].keys())
    # Sort: put O last
    classes = sorted(c for c in all_classes if c != "O") + (["O"] if "O" in all_classes else [])

    if not classes:
        return

    lw  = 40
    cw  = 10

    header = f"  {'Model + OCR':<{lw}}" + "".join(f"  {cls:>{cw}}" for cls in classes)
    sep    = "-" * len(header)
    print()
    print("=" * len(header))
    print("  PER-CLASS F1 BREAKDOWN")
    print("=" * len(header))
    print(header)
    print(sep)

    sorted_results = sorted(results, key=lambda r: r["overall"]["f1"], reverse=True)
    for r in sorted_results:
        lbl = _pair_label(r["model_dir"], r["ocr_engine"])
        row = f"  {lbl:<{lw}}"
        for cls in classes:
            f1 = r["per_class"].get(cls, {}).get("f1", None)
            cell = f"{f1:.4f}" if f1 is not None else "  n/a  "
            row += f"  {cell:>{cw}}"
        print(row)

    print("=" * len(header))


def _print_inferences(results: list):
    """Print model-selection inferences from the evaluation results."""
    sorted_r = sorted(results, key=lambda r: r["overall"]["f1"], reverse=True)
    best     = sorted_r[0]
    worst    = sorted_r[-1]

    # Group by base model
    funsd_results      = [r for r in results if r["model_dir"] == "layoutlmv3-funsd" and r["ocr_engine"] != "gt"]
    funsd_gt_result    = next((r for r in results if r["model_dir"] == "layoutlmv3-funsd" and r["ocr_engine"] == "gt"), None)
    ocr_specific       = [r for r in results if r["model_dir"] != "layoutlmv3-funsd"]

    print()
    print("=" * 60)
    print("  🔍  INFERENCES & RECOMMENDATIONS")
    print("=" * 60)

    # 1. Best overall
    print(f"""
1. BEST OVERALL
   → {_pair_label(best['model_dir'], best['ocr_engine'])}
     F1={best['overall']['f1']:.4f}  Precision={best['overall']['precision']:.4f}
     Recall={best['overall']['recall']:.4f}  Accuracy={best['overall']['accuracy']:.4f}
""")

    # 2. GT baseline
    if funsd_gt_result:
        print(f"""2. GT-BBOX BASELINE (layoutlmv3-funsd + ground-truth boxes)
   F1={funsd_gt_result['overall']['f1']:.4f}
   • This is the theoretical upper bound: the model was trained with
     perfect bounding boxes from annotations and is evaluated the same way.
   • Any OCR-based variant will score below this because OCR boxes differ
     from perfectly annotated GT boxes.
""")

    # 3. GT model vs OCRs
    if funsd_results:
        print("3. layoutlmv3-funsd (GT-trained) WITH DIFFERENT OCRs")
        print("   (Shows OCR impact on a model NOT trained with that OCR)")
        for r in sorted(funsd_results, key=lambda x: x["overall"]["f1"], reverse=True):
            delta = r["overall"]["f1"] - (funsd_gt_result["overall"]["f1"] if funsd_gt_result else 0)
            print(f"   {r['ocr_engine']:12s}: F1={r['overall']['f1']:.4f}  (Δ vs GT baseline: {delta:+.4f})")
        print("   • The OCR with the smallest F1 drop vs GT is the best match")
        print("     for the layoutlmv3-funsd model's spatial encoding.")
        print()

    # 4. Native OCR-model pairs
    if ocr_specific:
        print("4. NATIVE OCR-TRAINED MODELS")
        for r in sorted(ocr_specific, key=lambda x: x["overall"]["f1"], reverse=True):
            print(f"   {_pair_label(r['model_dir'], r['ocr_engine']):42s}: F1={r['overall']['f1']:.4f}")
        print()
        best_native = sorted(ocr_specific, key=lambda x: x["overall"]["f1"], reverse=True)[0]
        print(f"   → Best native model: {_pair_label(best_native['model_dir'], best_native['ocr_engine'])}")
        print("     Use this model when running on unseen documents without GT boxes.")
        print()

    # 5. Fine-tuning recommendation
    print("5. FINE-TUNING RECOMMENDATION")
    best_native = sorted(ocr_specific, key=lambda x: x["overall"]["f1"], reverse=True)[0] if ocr_specific else best
    print(f"   → Continue fine-tuning: {best_native['model_dir']}")
    print(f"      with OCR: {best_native['ocr_engine']}")
    print(f"      from checkpoint: {best_native['checkpoint']}")
    print()
    print("   Rationale:")
    print("   • The OCR-specific model is already adapted to real OCR noise.")
    print("   • Fine-tune on domain-specific forms to close the gap with the")
    print("     GT-bbox model without needing annotated bounding boxes.")

    print("=" * 60)
    print()


def _save_all_results(results: list, output_dir: str):
    """Save combined JSON."""
    combined = []
    for r in results:
        combined.append({
            "model_dir":       r["model_dir"],
            "ocr_engine":      r["ocr_engine"],
            "checkpoint":      r["checkpoint"],
            "overall":         r["overall"],
            "per_class":       r["per_class"],
            "total_docs":      r["total_docs"],
            "total_gt_words":  r["total_gt_words"],
            "total_ocr_words": r["total_ocr_words"],
            "elapsed_seconds": r["elapsed_seconds"],
        })
        # Per-pair file
        safe_name = f"layoutlm_{r['model_dir'].replace('layoutlmv3-', '')}_{r['ocr_engine']}"
        per_pair_path = os.path.join(output_dir, f"{safe_name}.json")
        with open(per_pair_path, "w") as f:
            json.dump(
                {"model_dir": r["model_dir"], "ocr_engine": r["ocr_engine"],
                 "checkpoint": r["checkpoint"], "overall": r["overall"],
                 "per_class": r["per_class"]},
                f, indent=2, cls=NumpyEncoder
            )
        print(f"  Saved → {per_pair_path}")

    combined_path = os.path.join(output_dir, "layoutlm_comparison.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, cls=NumpyEncoder)
    print(f"\nCombined JSON → {combined_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all LayoutLMv3 model × OCR pairs on FUNSD test split."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
        help=f"HuggingFace cache directory (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSON reports (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of test documents per pair (for smoke tests)"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        metavar="MODEL:OCR",
        help=(
            "Subset of pairs to evaluate, e.g. "
            "'funsd-tesseract:tesseract funsd:gt'. "
            "Use model short-name (drop 'layoutlmv3-' prefix)."
        )
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Filter pairs if requested
    pairs = ALL_PAIRS
    if args.pairs:
        wanted = set()
        for p in args.pairs:
            short, ocr = p.rsplit(":", 1)
            wanted.add((f"layoutlmv3-{short}", ocr))
        pairs = [p for p in ALL_PAIRS if p in wanted]
        if not pairs:
            print(f"ERROR: None of {args.pairs} matched known pairs.")
            sys.exit(1)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("  LAYOUTLM × OCR EVALUATION  (FUNSD test split)")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Cache dir   : {args.cache_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Limit       : {'none' if args.limit is None else args.limit}")
    print(f"  Pairs to run: {len(pairs)}")
    for m, o in pairs:
        print(f"    {_pair_label(m, o)}")
    print()

    print("[ Loading FUNSD dataset via HuggingFace ]")
    dataset = load_dataset("nielsr/funsd", cache_dir=args.cache_dir)
    print(f"  Train: {len(dataset['train'])}  Test: {len(dataset['test'])}\n")

    total_start = time.time()
    all_results = []

    for pair_idx, (model_dir, ocr_engine) in enumerate(pairs, start=1):
        label = _pair_label(model_dir, ocr_engine)
        print(f"\n{'=' * 60}")
        print(f"  [{pair_idx}/{len(pairs)}]  {label}")
        print(f"{'=' * 60}")

        t0 = time.time()
        try:
            result = evaluate_pair(
                model_dir=model_dir,
                ocr_engine=ocr_engine,
                dataset=dataset,
                cache_dir=args.cache_dir,
                limit=args.limit,
                device=device,
            )
        except Exception as e:
            print(f"  ❌  FAILED: {e}")
            all_results.append({
                "model_dir": model_dir, "ocr_engine": ocr_engine,
                "error": str(e),
                "overall": {"f1": 0, "precision": 0, "recall": 0, "accuracy": 0},
                "per_class": {}, "checkpoint": "N/A",
                "total_docs": 0, "total_gt_words": 0, "total_ocr_words": 0,
                "elapsed_seconds": round(time.time() - t0, 2),
                "seconds_per_doc": 0,
            })
            continue

        all_results.append(result)

        print(f"  ✓  F1={result['overall']['f1']:.4f}  "
              f"Prec={result['overall']['precision']:.4f}  "
              f"Recall={result['overall']['recall']:.4f}  "
              f"Acc={result['overall']['accuracy']:.4f}")
        print(f"  ⏱  {result['elapsed_seconds']:.1f}s  "
              f"({result['seconds_per_doc']:.2f}s/doc)")

        # Per-class snapshot
        if result["per_class"]:
            print("  Per-class F1:", {k: v["f1"] for k, v in result["per_class"].items()})

    # -----------------------------------------------------------------------
    # Save & report
    # -----------------------------------------------------------------------
    print("\n[ Saving results ]")
    _save_all_results(all_results, args.output_dir)

    # Filter out failed runs for display
    good_results = [r for r in all_results if "error" not in r]
    if good_results:
        _print_comparison_table(good_results)
        _print_per_class_table(good_results)
        _print_inferences(good_results)

    total_elapsed = time.time() - total_start
    print(f"⏱  Total wall-clock time: {total_elapsed:.1f}s")
    print(f"   Pairs completed: {len(good_results)} / {len(pairs)}")
    print()


if __name__ == "__main__":
    main()
