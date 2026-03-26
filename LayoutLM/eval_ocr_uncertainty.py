"""
eval_ocr_uncertainty.py
=======================
Evaluates uncertainty quality for token/entity predictions by combining:
- OCR confidence
- model confidence

Outputs a full JSON report with:
- overall seqeval metrics
- confidence-bin reliability stats
- confidence-accuracy curve
- entity-type OCR sensitivity
- uncertainty flagging summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset

try:
    import evaluate
except Exception:
    evaluate = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from editor.inference import LayoutLMGraphLinkingService
from LayoutLM.layoutlm_customOCR import align_ocr_to_ground_truth


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _build_service(checkpoint_dir: str | None) -> LayoutLMGraphLinkingService:
    if checkpoint_dir:
        ckpt = Path(checkpoint_dir).expanduser().resolve()
    else:
        ckpt = Path(SCRIPT_DIR) / "layoutlmv3-funsd" / "checkpoint-608"

    if ckpt.exists():
        return LayoutLMGraphLinkingService(checkpoint_dir=ckpt)

    # fallback to service defaults if explicit checkpoint is missing
    return LayoutLMGraphLinkingService()


def _confidence_bins(token_stats: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    entity_stats = [t for t in token_stats if t["is_entity"]]
    if not entity_stats:
        return {"bins": [], "ece": 0.0, "brier": 0.0}

    bins = []
    ece = 0.0
    total = len(entity_stats)
    brier = 0.0

    for t in entity_stats:
        p = float(t["overall_conf"])
        y = 1.0 if t["is_correct_entity"] else 0.0
        brier += (p - y) ** 2

    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        subset = [t for t in entity_stats if low <= t["overall_conf"] < high or (i == n_bins - 1 and t["overall_conf"] == 1.0)]
        if not subset:
            continue

        avg_conf = float(np.mean([t["overall_conf"] for t in subset]))
        avg_acc = float(np.mean([1.0 if t["is_correct_entity"] else 0.0 for t in subset]))
        weight = len(subset) / total
        ece += weight * abs(avg_acc - avg_conf)

        bins.append(
            {
                "low": round(low, 4),
                "high": round(high, 4),
                "count": len(subset),
                "avg_confidence": round(avg_conf, 6),
                "empirical_accuracy": round(avg_acc, 6),
                "gap": round(avg_acc - avg_conf, 6),
            }
        )

    return {
        "bins": bins,
        "ece": round(float(ece), 6),
        "brier": round(float(brier / max(total, 1)), 6),
    }


def _confidence_accuracy_curve(token_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entity_stats = [t for t in token_stats if t["is_entity"]]
    total = len(entity_stats)
    if total == 0:
        return []

    curve = []
    thresholds = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]
    for thr in thresholds:
        kept = [t for t in entity_stats if t["overall_conf"] >= thr]
        kept_n = len(kept)
        correct = sum(1 for t in kept if t["is_correct_entity"])
        acc = _safe_div(correct, kept_n)
        coverage = _safe_div(kept_n, total)
        curve.append(
            {
                "threshold": thr,
                "coverage": round(coverage, 6),
                "accuracy": round(acc, 6),
                "kept_entities": kept_n,
                "total_entities": total,
            }
        )
    return curve


def _entity_type_ocr_sensitivity(token_stats: List[Dict[str, Any]], low_thr: float) -> Dict[str, Dict[str, float]]:
    entity_types = ["QUESTION", "ANSWER", "HEADER"]
    out: Dict[str, Dict[str, float]] = {}

    for etype in entity_types:
        type_rows = [t for t in token_stats if t["is_entity"] and etype in str(t["true_label"]).upper()]
        high = [t for t in type_rows if t["ocr_conf"] >= low_thr]
        low = [t for t in type_rows if t["ocr_conf"] < low_thr]
        high_acc = _safe_div(sum(1 for t in high if t["is_correct_entity"]), len(high))
        low_acc = _safe_div(sum(1 for t in low if t["is_correct_entity"]), len(low))
        out[etype.lower()] = {
            "high_ocr_count": len(high),
            "low_ocr_count": len(low),
            "high_ocr_accuracy": round(high_acc, 6),
            "low_ocr_accuracy": round(low_acc, 6),
            "degradation": round(high_acc - low_acc, 6),
        }

    return out


def _extract_bio_entities(labels: Sequence[str]) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    start = -1
    ent_type = None

    def flush(end_idx: int):
        nonlocal start, ent_type
        if ent_type is not None and start >= 0:
            out.append((ent_type, start, end_idx))
        start = -1
        ent_type = None

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            flush(i - 1)
            ent_type = label[2:]
            start = i
            continue

        if label.startswith("I-"):
            cur_type = label[2:]
            if ent_type == cur_type and start >= 0:
                continue
            flush(i - 1)
            ent_type = cur_type
            start = i
            continue

        flush(i - 1)

    flush(len(labels) - 1)
    return out


def _fallback_seqeval(pred_labels: Sequence[Sequence[str]], true_labels: Sequence[Sequence[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    correct_tokens = 0
    total_tokens = 0

    for doc_idx, (pred_doc, true_doc) in enumerate(zip(pred_labels, true_labels)):
        length = min(len(pred_doc), len(true_doc))
        if length <= 0:
            continue
        pred_doc = list(pred_doc[:length])
        true_doc = list(true_doc[:length])

        total_tokens += length
        correct_tokens += sum(1 for p, t in zip(pred_doc, true_doc) if p == t)

        pred_entities = {(doc_idx, ent_type, s, e) for ent_type, s, e in _extract_bio_entities(pred_doc)}
        true_entities = {(doc_idx, ent_type, s, e) for ent_type, s, e in _extract_bio_entities(true_doc)}
        tp += len(pred_entities & true_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float(correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    return {
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "overall_accuracy": accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR/model uncertainty calibration on FUNSD.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional datasets cache dir")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Optional LayoutLM checkpoint dir")
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(SCRIPT_DIR, "eval_results", "uncertainty_calibration.json"),
        help="Output JSON path",
    )
    parser.add_argument("--low_ocr_threshold", type=float, default=0.6, help="Low OCR confidence threshold")
    parser.add_argument("--limit", type=int, default=None, help="Optional doc limit (default: full test split)")
    parser.add_argument(
        "--ocr_source",
        type=str,
        default="gt",
        choices=["gt", "service"],
        help="Use GT words/boxes (gt) or service OCR (service).",
    )
    args = parser.parse_args()

    dataset = load_dataset("nielsr/funsd", cache_dir=args.cache_dir)
    test_split = dataset["test"]
    if args.limit is not None:
        test_split = test_split.select(range(min(int(args.limit), len(test_split))))

    service = _build_service(args.checkpoint_dir)
    label_list = [service.id2label[i] for i in range(len(service.id2label))]
    o_label_id = service.token_model.config.label2id.get("O", 0)

    true_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []
    token_stats: List[Dict[str, Any]] = []

    for idx, example in enumerate(test_split):
        if idx % 10 == 0:
            print(f"[uncertainty] processing {idx}/{len(test_split)}")

        image = example["image"].convert("RGB")
        gt_words = example["words"]
        gt_boxes = example["bboxes"]
        gt_labels = example["ner_tags"]

        if args.ocr_source == "gt":
            words = list(gt_words)
            boxes = [list(map(int, b)) for b in gt_boxes]
            ocr_confs = [1.0] * len(words)
        else:
            words, boxes, ocr_confs = service._run_ocr(image)
        if not words:
            continue

        aligned_ids = align_ocr_to_ground_truth(
            words, boxes, gt_words, gt_boxes, gt_labels, default_label_id=o_label_id
        )
        aligned_labels = [label_list[lid] for lid in aligned_ids]
        preds = service._predict_word_labels(image, words, boxes, ocr_confs)

        doc_true: List[str] = []
        doc_pred: List[str] = []

        for i, wp in enumerate(preds):
            true_label = aligned_labels[i]
            pred_label = wp.label
            ocr_conf = float(wp.ocr_confidence)
            if ocr_conf > 1.0:
                ocr_conf /= 100.0  # tesseract style 0..100
            model_conf = float(wp.model_confidence)
            overall_conf = float(np.clip(ocr_conf * model_conf, 0.0, 1.0))

            doc_true.append(true_label)
            doc_pred.append(pred_label)

            token_stats.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "ocr_conf": ocr_conf,
                    "model_conf": model_conf,
                    "overall_conf": overall_conf,
                    "is_entity": (true_label != "O" or pred_label != "O"),
                    "is_correct_entity": (true_label == pred_label and true_label != "O"),
                }
            )

        true_labels.append(doc_true)
        pred_labels.append(doc_pred)

    if evaluate is not None:
        try:
            seqeval = evaluate.load("seqeval")
            overall = seqeval.compute(predictions=pred_labels, references=true_labels)
            seqeval_backend = "evaluate/seqeval"
        except Exception:
            overall = _fallback_seqeval(pred_labels, true_labels)
            seqeval_backend = "builtin_fallback_seqeval"
    else:
        overall = _fallback_seqeval(pred_labels, true_labels)
        seqeval_backend = "builtin_fallback_seqeval"

    reliability = _confidence_bins(token_stats, n_bins=10)
    curve = _confidence_accuracy_curve(token_stats)
    ocr_sensitivity = _entity_type_ocr_sensitivity(token_stats, low_thr=float(args.low_ocr_threshold))

    entity_stats = [t for t in token_stats if t["is_entity"]]
    flagged = [t for t in entity_stats if t["overall_conf"] < float(args.low_ocr_threshold)]

    payload = {
        "status": "ok",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": "nielsr/funsd",
        "split": "test",
        "ocr_source": args.ocr_source,
        "seqeval_backend": seqeval_backend,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else "default_service_checkpoint",
        "overall_seqeval": {
            "precision": round(float(overall.get("overall_precision", 0.0)), 6),
            "recall": round(float(overall.get("overall_recall", 0.0)), 6),
            "f1": round(float(overall.get("overall_f1", 0.0)), 6),
            "accuracy": round(float(overall.get("overall_accuracy", 0.0)), 6),
        },
        "counts": {
            "docs_evaluated": len(true_labels),
            "tokens_total": len(token_stats),
            "entity_tokens_total": len(entity_stats),
            "flagged_low_conf_entity_tokens": len(flagged),
            "low_conf_threshold": float(args.low_ocr_threshold),
        },
        "calibration": reliability,
        "confidence_accuracy_curve": curve,
        "entity_type_ocr_sensitivity": ocr_sensitivity,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({"status": "ok", "output_path": args.output_path}, indent=2))


if __name__ == "__main__":
    main()
