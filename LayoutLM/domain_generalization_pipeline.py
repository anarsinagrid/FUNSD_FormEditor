"""
domain_generalization_pipeline.py
================================
Standalone OOD generalization + concept steering pipeline for LayoutLMv3.

Task scope:
- No fine-tuning of model weights.
- Phase 1 (required now): RVL-CDIP OOD evaluation.
- Test-time concept steering using four concepts:
    handwritten, printed, structured, unstructured

Subcommands:
  - prepare_data
  - extract_concepts
  - eval_baseline
  - eval_steered
  - transfer_matrix
  - run_all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import evaluate as hf_evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from layoutlm_customOCR import OCRBackend, align_ocr_to_ground_truth

DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "layoutlmv3-funsd-doctr", "checkpoint-1250")
DEFAULT_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, ".hf_cache")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "domain_generalization_artifacts")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "domain_transfer_report.json")
DEFAULT_CONCEPT_VECTOR_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "concept_vectors.pt")
DEFAULT_CONCEPT_STATS_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "concept_vector_stats.json")
DEFAULT_MATRIX_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "transfer_matrix.csv")
DEFAULT_INVARIANCE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "entity_invariance_report.md")

CONCEPTS = ["handwritten", "printed", "structured", "unstructured"]
FUNSD_DATASET_NAME = "nielsr/funsd"
RVL_CANDIDATES = ["chainyo/rvl-cdip", "rvl_cdip", "aharley/rvl_cdip"]
CORD_CANDIDATES = ["naver-clova-ix/cord-v2", "naver-ai-corp/cord-v2", "naver-ai/cord-v2", "naver-ai-corp--cord-v2"]


@dataclass
class NERPreparedSample:
    inputs: Dict[str, torch.Tensor]
    token_labels: torch.Tensor
    word_ids: List[Optional[int]]
    words: List[str]
    boxes_1000: List[List[int]]
    confidences: List[float]
    concept_scores: Dict[str, float]


@dataclass
class OCRDoc:
    words: List[str]
    boxes_1000: List[List[int]]
    confidences: List[float]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {v}")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_short() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            value = proc.stdout.strip()
            return value or None
    except Exception:
        return None
    return None


def _build_provenance(args, artifact_name: str) -> Dict[str, Any]:
    script_path = os.path.abspath(__file__)
    provenance = {
        "artifact_name": artifact_name,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "command": str(getattr(args, "command", "unknown")),
        "script_path": script_path,
        "script_sha256": _file_sha256(script_path),
        "git_commit_short": _git_commit_short(),
        "policy": {
            "ocr_inference": "doctr",
            "funsd_eval_docs": 50,
            "confidence_aware_e2e": "disabled_by_design",
        },
    }
    return provenance


def _attach_provenance(payload: Dict[str, Any], args, artifact_name: str) -> Dict[str, Any]:
    payload = dict(payload)
    payload["provenance"] = _build_provenance(args, artifact_name=artifact_name)
    return payload


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        return torch.device("mps")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")

    return torch.device("cpu")


def _resolve_path(path: str, default: Optional[str] = None) -> str:
    chosen = path or default
    if chosen is None:
        raise ValueError("Missing required path.")
    chosen = os.path.expanduser(chosen)
    if os.path.isabs(chosen):
        return chosen
    return os.path.abspath(os.path.join(REPO_ROOT, chosen))


def _resolve_cache_dir(cache_dir: str) -> str:
    invalid_prefixes = ("/absolute", "<absolute", "ABSOLUTE_PATH")
    if any(str(cache_dir).startswith(p) for p in invalid_prefixes):
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _read_input_size(checkpoint_dir: str) -> int:
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(cfg_path):
        return 384
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("input_size", 384))
    except Exception:
        return 384


def _pad_to_square(image: Image.Image, target: int) -> Image.Image:
    w, h = image.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    canvas.paste(image.resize((nw, nh), Image.LANCZOS), ((target - nw) // 2, (target - nh) // 2))
    return canvas


def _max_word_id(word_ids: Sequence[Optional[int]]) -> int:
    best = -1
    for wid in word_ids:
        if wid is not None and wid > best:
            best = int(wid)
    return best


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_artifact_path(output_dir: str, user_path: Optional[str], filename: str) -> str:
    if user_path:
        return _resolve_path(user_path)
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# Model and processor
# ---------------------------------------------------------------------------

def _load_model_and_processor(
    checkpoint_path: str,
    cache_dir: str,
    device: torch.device,
) -> Tuple[LayoutLMv3ForTokenClassification, LayoutLMv3Processor, int]:
    input_size = _read_input_size(checkpoint_path)

    try:
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(checkpoint_path, local_files_only=True)
        processor = LayoutLMv3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            apply_ocr=False,
        )

    processor.image_processor.size = {"height": input_size, "width": input_size}
    processor.image_processor.do_resize = True
    processor.image_processor.do_pad = True

    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "init_visual_bbox"):
        try:
            grid = input_size // 16
            model.layoutlmv3.init_visual_bbox(image_size=(grid, grid))
        except Exception:
            pass

    return model, processor, input_size


# ---------------------------------------------------------------------------
# OCR + concept feature extraction
# ---------------------------------------------------------------------------

def _run_ocr_doctr(backend: OCRBackend, image: Image.Image) -> OCRDoc:
    words, boxes_1000, confidences, _ = backend.run(image)
    if not words:
        words = [""]
        boxes_1000 = [[0, 0, 0, 0]]
        confidences = [0.0]
    return OCRDoc(
        words=[str(w) for w in words],
        boxes_1000=[list(map(int, b)) for b in boxes_1000],
        confidences=[float(c) for c in confidences] if confidences else [0.0] * len(words),
    )


def _bbox_center(box: Sequence[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _alignment_score(values: Sequence[float], bin_size: float = 25.0) -> float:
    if not values:
        return 0.0
    buckets: Dict[int, int] = {}
    for v in values:
        k = int(v // bin_size)
        buckets[k] = buckets.get(k, 0) + 1
    top = max(buckets.values()) if buckets else 0
    return float(top / max(1, len(values)))


def _compute_concept_scores(
    words: Sequence[str],
    boxes_1000: Sequence[Sequence[int]],
    confidences: Sequence[float],
) -> Dict[str, float]:
    n = len(words)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    avg_conf = min(max(avg_conf, 0.0), 1.0)

    centers = [_bbox_center(b) for b in boxes_1000] if boxes_1000 else []
    ys = [c[1] for c in centers]
    xs = [c[0] for c in centers]

    row_align = _alignment_score(ys, bin_size=25.0)
    col_align = _alignment_score(xs, bin_size=30.0)

    count_score = min(float(n) / 120.0, 1.0)

    structured = 0.35 * row_align + 0.25 * col_align + 0.20 * count_score + 0.20 * avg_conf
    structured = float(min(max(structured, 0.0), 1.0))
    unstructured = float(min(max(1.0 - structured, 0.0), 1.0))

    handwriting_noise = 0.60 * (1.0 - avg_conf) + 0.40 * (1.0 - row_align)
    handwritten = float(min(max(handwriting_noise, 0.0), 1.0))
    printed = float(min(max(1.0 - handwritten, 0.0), 1.0))

    raw = {
        "handwritten": handwritten,
        "printed": printed,
        "structured": structured,
        "unstructured": unstructured,
    }
    total = sum(raw.values()) + 1e-8
    return {k: float(v / total) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Forward pass with optional concept steering
# ---------------------------------------------------------------------------

def _concept_weighted_vector(
    concept_vectors: Optional[Dict[str, torch.Tensor]],
    concept_scores: Dict[str, float],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if concept_vectors is None:
        return None
    vecs = []
    for concept in CONCEPTS:
        if concept not in concept_vectors:
            continue
        vec = concept_vectors[concept].to(device)
        vecs.append(float(concept_scores.get(concept, 0.0)) * vec)
    if not vecs:
        return None
    out = torch.stack(vecs, dim=0).sum(dim=0)
    return out


def _forward_sequence_output(
    model: LayoutLMv3ForTokenClassification,
    inputs: Dict[str, torch.Tensor],
    concept_vectors: Optional[Dict[str, torch.Tensor]],
    concept_scores: Dict[str, float],
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "bbox": inputs["bbox"],
        "pixel_values": inputs["pixel_values"],
        "return_dict": True,
    }
    if "token_type_ids" in inputs:
        model_inputs["token_type_ids"] = inputs["token_type_ids"]

    outputs = model.layoutlmv3(**model_inputs)
    seq_len = int(inputs["input_ids"].shape[1])
    sequence_output = outputs[0][:, :seq_len, :]

    steering_vec = _concept_weighted_vector(
        concept_vectors=concept_vectors,
        concept_scores=concept_scores,
        device=sequence_output.device,
    )
    if steering_vec is not None and float(alpha) != 0.0:
        sequence_output = sequence_output + float(alpha) * steering_vec.view(1, 1, -1)

    attn = inputs["attention_mask"].to(sequence_output.dtype)
    denom = attn.sum(dim=1, keepdim=True).clamp(min=1.0)
    doc_vec = (sequence_output * attn.unsqueeze(-1)).sum(dim=1) / denom

    logits = model.classifier(model.dropout(sequence_output))
    return logits, doc_vec


# ---------------------------------------------------------------------------
# FUNSD (source / entity metrics)
# ---------------------------------------------------------------------------

def _prepare_funsd_samples(
    split,
    processor: LayoutLMv3Processor,
    ocr_backend: OCRBackend,
    o_label_id: int,
    input_size: int,
    limit: Optional[int] = None,
) -> Tuple[List[NERPreparedSample], Dict[str, Any]]:
    if limit is not None:
        split = split.select(range(min(int(limit), len(split))))

    samples: List[NERPreparedSample] = []
    trunc_hits = 0

    for idx, ex in enumerate(split):
        print(f"[prepare_funsd] {idx + 1}/{len(split)}", end="\r")

        image = ex["image"].convert("RGB")
        gt_words = ex["words"]
        gt_boxes = ex["bboxes"]
        gt_labels = ex["ner_tags"]

        ocr = _run_ocr_doctr(ocr_backend, image)
        concept_scores = _compute_concept_scores(ocr.words, ocr.boxes_1000, ocr.confidences)

        aligned_labels = align_ocr_to_ground_truth(
            ocr.words,
            ocr.boxes_1000,
            gt_words,
            gt_boxes,
            gt_labels,
            default_label_id=o_label_id,
        )

        image_sq = _pad_to_square(image, input_size)
        enc = processor(
            image_sq,
            ocr.words,
            boxes=ocr.boxes_1000,
            word_labels=aligned_labels,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        max_wid = _max_word_id(word_ids)
        kept_words = max_wid + 1 if max_wid >= 0 else 0
        if kept_words < len(ocr.words):
            trunc_hits += 1

        inputs: Dict[str, torch.Tensor] = {}
        for k in ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"):
            if k in enc:
                inputs[k] = enc[k]

        samples.append(
            NERPreparedSample(
                inputs=inputs,
                token_labels=enc["labels"].squeeze(0).to(torch.long),
                word_ids=list(word_ids),
                words=ocr.words,
                boxes_1000=ocr.boxes_1000,
                confidences=ocr.confidences,
                concept_scores=concept_scores,
            )
        )

    print()
    return samples, {"docs": len(samples), "truncation_hits": int(trunc_hits)}


def _collapse_word_predictions(
    pred_ids: Sequence[int],
    token_labels: Sequence[int],
    word_ids: Sequence[Optional[int]],
    label_list: Sequence[str],
    o_label_id: int,
) -> Tuple[List[str], List[str], List[int]]:
    max_wid = _max_word_id(word_ids)
    n_words = max_wid + 1 if max_wid >= 0 else 0
    word_pred_ids = [int(o_label_id)] * max(0, n_words)

    doc_true: List[str] = []
    doc_pred: List[str] = []

    prev = None
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid == prev:
            continue
        prev = wid

        if wid < 0 or wid >= len(word_pred_ids):
            continue

        true_lid = int(token_labels[pos])
        pred_lid = int(pred_ids[pos])
        word_pred_ids[wid] = pred_lid

        if true_lid != -100:
            doc_true.append(label_list[true_lid])
            doc_pred.append(label_list[pred_lid])

    return doc_true, doc_pred, word_pred_ids


def _evaluate_funsd_entity(
    model: LayoutLMv3ForTokenClassification,
    samples: Sequence[NERPreparedSample],
    label_list: Sequence[str],
    cache_dir: str,
    concept_vectors: Optional[Dict[str, torch.Tensor]],
    alpha: float,
) -> Dict[str, Any]:
    seqeval = hf_evaluate.load("seqeval", cache_dir=cache_dir)

    true_all: List[List[str]] = []
    pred_all: List[List[str]] = []
    latencies: List[float] = []

    label2id = {v: i for i, v in enumerate(label_list)}
    o_label_id = int(label2id.get("O", 0))

    for sample in samples:
        inputs = {k: v.to(next(model.parameters()).device) for k, v in sample.inputs.items()}
        token_labels = sample.token_labels.cpu().tolist()

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = _forward_sequence_output(
                model=model,
                inputs=inputs,
                concept_vectors=concept_vectors,
                concept_scores=sample.concept_scores,
                alpha=alpha,
            )
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred_ids = logits.argmax(-1).squeeze(0).cpu().tolist()
        doc_true, doc_pred, _ = _collapse_word_predictions(
            pred_ids=pred_ids,
            token_labels=token_labels,
            word_ids=sample.word_ids,
            label_list=label_list,
            o_label_id=o_label_id,
        )
        true_all.append(doc_true)
        pred_all.append(doc_pred)

    result = seqeval.compute(predictions=pred_all, references=true_all)
    overall = {
        "precision": round(float(result["overall_precision"]), 4),
        "recall": round(float(result["overall_recall"]), 4),
        "f1": round(float(result["overall_f1"]), 4),
        "accuracy": round(float(result["overall_accuracy"]), 4),
    }

    per_class = {}
    for k, v in result.items():
        if isinstance(v, dict):
            per_class[k] = {
                "precision": round(float(v.get("precision", 0.0)), 4),
                "recall": round(float(v.get("recall", 0.0)), 4),
                "f1": round(float(v.get("f1", 0.0)), 4),
                "number": int(v.get("number", 0)),
            }

    return {
        "metric_type": "entity_f1",
        "overall": overall,
        "token_f1": _calculate_token_level_f1(true_all, pred_all),
        "per_class": per_class,
        "avg_model_latency_ms": round(float(statistics.mean(latencies)) if latencies else 0.0, 3),
        "docs": len(samples),
    }


def _calculate_token_level_f1(true_all: List[List[str]], pred_all: List[List[str]]) -> float:
    """Computes Micro-F1 at the token level, disregarding BIO sequence semantics."""
    t_labels = [l for seq in true_all for l in seq]
    p_labels = [l for seq in pred_all for l in seq]
    
    tp = 0
    fp = 0
    fn = 0
    
    for t, p in zip(t_labels, p_labels):
        if t == p:
            if t != "O":
                tp += 1
        else:
            if t != "O":
                fn += 1
            if p != "O":
                fp += 1
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(float(f1), 4)


# ---------------------------------------------------------------------------
# RVL-CDIP (classification transfer)
# ---------------------------------------------------------------------------

def _load_rvl_dataset(cache_dir: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    import glob
    errors = {}

    # Try to load local parquet files explicitly
    parquet_pattern = os.path.join(cache_dir, "datasets", "chainyo___rvl-cdip", "default", "*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    if not parquet_files:
        parquet_pattern = os.path.join(cache_dir, "datasets", "chainyo--rvl-cdip", "default", "*.parquet")
        parquet_files = glob.glob(parquet_pattern)

    if parquet_files:
        try:
            ds_test = load_dataset("parquet", data_files={"test": parquet_files}, split="test", cache_dir=cache_dir)
            from datasets import Image as DatasetsImage
            ds_test = ds_test.cast_column("image", DatasetsImage())
            ds = {"test": ds_test}
            return ds, {"status": "ready", "dataset_name": "rvl_cdip_local_parquet", "splits": ["test"]}
        except Exception as e:
            errors["local_parquet"] = str(e)
            print(f"[warning] Failed to load local parquet files for RVL-CDIP: {e}")

    for name in RVL_CANDIDATES:
        try:
            # Try to load with split='test' first (faster when cached)
            try:
                ds_test = load_dataset(name, split="test", cache_dir=cache_dir)
                # Wrap in dict format for consistent interface
                ds = {"test": ds_test}
                return ds, {"status": "ready", "dataset_name": name, "splits": ["test"]}
            except Exception as e_split:
                # Fallback: load entire dataset without split specification
                ds = load_dataset(name, cache_dir=cache_dir)
                if not hasattr(ds, 'keys'):
                    # Single dataset, wrap it
                    ds = {"test": ds}
                return ds, {"status": "ready", "dataset_name": name, "splits": list(ds.keys())}
        except Exception as e:
            errors[name] = str(e)
    return None, {
        "status": "error",
        "reason": "Unable to load RVL-CDIP from known dataset names.",
        "tried": errors,
        "action": "Check internet / HF access. If needed, manually download and mount a local RVL dataset.",
    }


def _load_cord_dataset(cache_dir: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    import glob
    errors = {}
    for name in CORD_CANDIDATES:
        try:
            ds = load_dataset(name, cache_dir=cache_dir)
            return ds, {"status": "ready", "dataset_name": name}
        except Exception as e:
            errors[name] = str(e)

    # Try local parquet fallback
    parquet_pattern = os.path.join(cache_dir, "datasets", "naver-ai-corp___cord-v2", "default", "*.parquet")
    parquet_files = glob.glob(parquet_pattern)
    if not parquet_files:
        parquet_pattern = os.path.join(cache_dir, "datasets", "naver-ai-corp--cord-v2", "default", "*.parquet")
        parquet_files = glob.glob(parquet_pattern)

    if parquet_files:
        try:
            ds = load_dataset("parquet", data_files={"test": parquet_files}, cache_dir=cache_dir)
            return ds, {"status": "ready", "source": "local_parquet", "files": len(parquet_files)}
        except Exception as e:
            errors["local_parquet"] = str(e)

    return None, {
        "status": "error",
        "reason": "Unable to load CORD from known dataset names.",
        "tried": errors,
        "action": "Check internet / HF access. For offline mode, ensure CORD is in cache_dir.",
    }


def _load_funsd_dataset(cache_dir: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    try:
        ds = load_dataset(FUNSD_DATASET_NAME, cache_dir=cache_dir)
        return ds, {"status": "ready", "dataset_name": FUNSD_DATASET_NAME}
    except Exception as e:
        return None, {
            "status": "error",
            "reason": "Unable to load FUNSD source dataset.",
            "tried": {FUNSD_DATASET_NAME: str(e)},
            "action": (
                "Check internet / HF access and local cache permissions. "
                "If this environment is offline, pre-download FUNSD and point cache_dir to that cache."
            ),
        }


def _rvl_label_names(dataset) -> Dict[int, str]:
    try:
        names = dataset["train"].features["label"].names
        return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        labels = set()
        for split_name in dataset.keys():
            split = dataset[split_name]
            for i in range(min(1000, len(split))):
                labels.add(int(split[i]["label"]))
        return {i: f"label_{i}" for i in sorted(labels)}


def _prepare_doc_inputs_for_image(
    image: Image.Image,
    processor: LayoutLMv3Processor,
    ocr_backend: OCRBackend,
    input_size: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], OCRDoc]:
    ocr = _run_ocr_doctr(ocr_backend, image)
    concept_scores = _compute_concept_scores(ocr.words, ocr.boxes_1000, ocr.confidences)

    image_sq = _pad_to_square(image, input_size)
    enc = processor(
        image_sq,
        ocr.words,
        boxes=ocr.boxes_1000,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    inputs = {}
    for k in ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"):
        if k in enc:
            inputs[k] = enc[k]
    return inputs, concept_scores, ocr


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(p=2).clamp(min=1e-8)


def _build_rvl_prototypes(
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    dataset_split,
    input_size: int,
    ocr_backend: OCRBackend,
    train_limit: Optional[int],
) -> Dict[int, torch.Tensor]:
    if train_limit is not None:
        dataset_split = dataset_split.select(range(min(int(train_limit), len(dataset_split))))

    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    for idx, ex in enumerate(dataset_split):
        print(f"[rvl_proto] {idx + 1}/{len(dataset_split)}", end="\r")
        image = ex["image"].convert("RGB")
        label = int(ex["label"])

        inputs_cpu, concept_scores, _ = _prepare_doc_inputs_for_image(
            image=image,
            processor=processor,
            ocr_backend=ocr_backend,
            input_size=input_size,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs_cpu.items()}

        with torch.no_grad():
            _, doc_vec = _forward_sequence_output(
                model=model,
                inputs=inputs,
                concept_vectors=None,
                concept_scores=concept_scores,
                alpha=0.0,
            )
        vec = _normalize(doc_vec.squeeze(0).detach().cpu())

        if label not in sums:
            sums[label] = vec.clone()
            counts[label] = 1
        else:
            sums[label] += vec
            counts[label] += 1

    print()

    prototypes = {}
    for label, vec_sum in sums.items():
        prototypes[label] = _normalize(vec_sum / max(1, counts[label]))
    return prototypes


def _build_cord_prototypes(
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    dataset_split,
    input_size: int,
    ocr_backend: OCRBackend,
    train_limit: Optional[int],
) -> Dict[str, torch.Tensor]:
    """CORD uses string labels in its 'ground_truth' or 'label' field depending on version."""
    if train_limit is not None:
        dataset_split = dataset_split.select(range(min(int(train_limit), len(dataset_split))))

    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for idx, ex in enumerate(dataset_split):
        print(f"[cord_proto] {idx + 1}/{len(dataset_split)}", end="\r")
        image = ex["image"].convert("RGB")
        
        # CORD-v2 usually has a 'label' or we use some document-level proxy.
        # If it's pure receipts, we can use the main categories if available.
        # For OOD, we'll try to extract the most descriptive class.
        label = "receipt" # Default fallback
        if "label" in ex:
            label = str(ex["label"])
        elif "ground_truth" in ex:
            # Try to find a coarse category in JSON
            try:
                gt = json.loads(ex["ground_truth"])
                label = gt.get("valid_line", [{}])[0].get("category", "receipt")
            except: pass

        inputs_cpu, concept_scores, _ = _prepare_doc_inputs_for_image(
            image=image,
            processor=processor,
            ocr_backend=ocr_backend,
            input_size=input_size,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs_cpu.items()}

        with torch.no_grad():
            _, doc_vec = _forward_sequence_output(
                model=model,
                inputs=inputs,
                concept_vectors=None,
                concept_scores=concept_scores,
                alpha=0.0,
            )
        vec = _normalize(doc_vec.squeeze(0).detach().cpu())

        if label not in sums:
            sums[label] = vec.clone()
            counts[label] = 1
        else:
            sums[label] += vec
            counts[label] += 1

    print()
    prototypes = {l: _normalize(s / max(1, counts[l])) for l, s in sums.items()}
    return prototypes


def _evaluate_cord(
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    dataset_split,
    input_size: int,
    ocr_backend: OCRBackend,
    prototypes: Dict[str, torch.Tensor],
    concept_vectors: Optional[Dict[str, torch.Tensor]],
    alpha: float,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if not prototypes:
        return {"status": "error", "reason": "No prototypes for CORD."}

    if limit is not None:
        dataset_split = dataset_split.select(range(min(int(limit), len(dataset_split))))

    correct = 0
    total = 0
    latencies: List[float] = []

    proto_keys = list(prototypes.keys())
    proto_matrix = torch.stack([prototypes[k] for k in proto_keys])

    for idx, ex in enumerate(dataset_split):
        image = ex["image"].convert("RGB")
        true_label = "receipt"
        if "label" in ex:
            true_label = str(ex["label"])
        elif "ground_truth" in ex:
            try:
                gt = json.loads(ex["ground_truth"])
                true_label = gt.get("valid_line", [{}])[0].get("category", "receipt")
            except: pass

        inputs_cpu, concept_scores, _ = _prepare_doc_inputs_for_image(
            image=image,
            processor=processor,
            ocr_backend=ocr_backend,
            input_size=input_size,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs_cpu.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            _, doc_vec = _forward_sequence_output(
                model=model,
                inputs=inputs,
                concept_vectors=concept_vectors,
                concept_scores=concept_scores,
                alpha=alpha,
            )
        latencies.append((time.perf_counter() - t0) * 1000.0)

        vec = _normalize(doc_vec.squeeze(0).detach().cpu())
        sims = torch.matmul(proto_matrix, vec)
        pred_idx = int(sims.argmax())
        pred_label = proto_keys[pred_idx]

        if pred_label == true_label:
            correct += 1
        total += 1

    accuracy = float(correct / total) if total > 0 else 0.0
    return {
        "status": "ok",
        "metric_type": "accuracy",
        "accuracy": round(accuracy, 4),
        "total_docs": total,
        "avg_latency_ms": round(float(statistics.mean(latencies)) if latencies else 0.0, 3)
    }


def _macro_f1_from_confusion(
    tp: Dict[int, int],
    fp: Dict[int, int],
    fn: Dict[int, int],
    labels: Sequence[int],
) -> float:
    f1s = []
    for label in labels:
        t = tp.get(label, 0)
        p = fp.get(label, 0)
        n = fn.get(label, 0)
        prec = t / (t + p) if (t + p) > 0 else 0.0
        rec = t / (t + n) if (t + n) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


def _evaluate_rvl(
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    dataset_split,
    input_size: int,
    ocr_backend: OCRBackend,
    prototypes: Dict[int, torch.Tensor],
    concept_vectors: Optional[Dict[str, torch.Tensor]],
    alpha: float,
    limit: Optional[int],
) -> Dict[str, Any]:
    if limit is not None:
        dataset_split = dataset_split.select(range(min(int(limit), len(dataset_split))))

    proto_labels = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[l] for l in proto_labels], dim=0)

    correct = 0
    total = 0
    tp: Dict[int, int] = {}
    fp: Dict[int, int] = {}
    fn: Dict[int, int] = {}
    latencies: List[float] = []

    for idx, ex in enumerate(dataset_split):
        print(f"[rvl_eval] {idx + 1}/{len(dataset_split)}", end="\r")
        image = ex["image"].convert("RGB")
        true_label = int(ex["label"])

        inputs_cpu, concept_scores, _ = _prepare_doc_inputs_for_image(
            image=image,
            processor=processor,
            ocr_backend=ocr_backend,
            input_size=input_size,
        )
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs_cpu.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            _, doc_vec = _forward_sequence_output(
                model=model,
                inputs=inputs,
                concept_vectors=concept_vectors,
                concept_scores=concept_scores,
                alpha=alpha,
            )
        latencies.append((time.perf_counter() - t0) * 1000.0)

        dv = _normalize(doc_vec.squeeze(0).detach().cpu())
        sims = torch.mv(proto_matrix, dv)
        pred_idx = int(torch.argmax(sims).item())
        pred_label = int(proto_labels[pred_idx])

        total += 1
        if pred_label == true_label:
            correct += 1
            tp[true_label] = tp.get(true_label, 0) + 1
        else:
            fp[pred_label] = fp.get(pred_label, 0) + 1
            fn[true_label] = fn.get(true_label, 0) + 1

    print()

    acc = float(correct / max(1, total))
    macro_f1 = _macro_f1_from_confusion(tp=tp, fp=fp, fn=fn, labels=proto_labels)

    return {
        "metric_type": "accuracy",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "docs": int(total),
        "avg_model_latency_ms": round(float(statistics.mean(latencies)) if latencies else 0.0, 3),
    }




# ---------------------------------------------------------------------------
# Concept vectors
# ---------------------------------------------------------------------------

def _extract_concept_vectors(
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    dataset,
    input_size: int,
    ocr_backend: OCRBackend,
    limit: Optional[int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    split = dataset["train"]
    if limit is not None:
        split = split.select(range(min(int(limit), len(split))))

    hidden_size = int(model.config.hidden_size)
    sums = {c: torch.zeros(hidden_size, dtype=torch.float32) for c in CONCEPTS}
    weights = {c: 0.0 for c in CONCEPTS}

    for idx, ex in enumerate(split):
        print(f"[extract_concepts] {idx + 1}/{len(split)}", end="\r")

        image = ex["image"].convert("RGB")
        ocr = _run_ocr_doctr(ocr_backend, image)
        concept_scores = _compute_concept_scores(ocr.words, ocr.boxes_1000, ocr.confidences)

        image_sq = _pad_to_square(image, input_size)
        enc = processor(
            image_sq,
            ocr.words,
            boxes=ocr.boxes_1000,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: enc[k].to(next(model.parameters()).device) for k in enc.keys()}

        with torch.no_grad():
            _, doc_vec = _forward_sequence_output(
                model=model,
                inputs=inputs,
                concept_vectors=None,
                concept_scores=concept_scores,
                alpha=0.0,
            )
        vec = doc_vec.squeeze(0).detach().cpu().to(torch.float32)

        for c in CONCEPTS:
            w = float(concept_scores.get(c, 0.0))
            sums[c] += w * vec
            weights[c] += w

    print()

    concept_vectors: Dict[str, torch.Tensor] = {}
    concept_stats = {"weights": {}, "norms": {}}
    for c in CONCEPTS:
        if weights[c] <= 0:
            concept_vectors[c] = torch.zeros(hidden_size, dtype=torch.float32)
            concept_stats["weights"][c] = 0.0
            concept_stats["norms"][c] = 0.0
            continue

        v = sums[c] / float(weights[c])
        v = v / v.norm(p=2).clamp(min=1e-8)
        concept_vectors[c] = v
        concept_stats["weights"][c] = round(float(weights[c]), 6)
        concept_stats["norms"][c] = round(float(v.norm(p=2).item()), 6)

    concept_stats["docs"] = len(split)
    return concept_vectors, concept_stats


def _save_concept_vectors(path: str, concept_vectors: Dict[str, torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in concept_vectors.items()}, path)


def _load_concept_vectors(path: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    out: Dict[str, torch.Tensor] = {}
    for c in CONCEPTS:
        if c in payload:
            out[c] = payload[c].to(torch.float32)
    if not out:
        raise RuntimeError(f"No concept vectors found in {path}")
    return out


# ---------------------------------------------------------------------------
# Matrix + invariance reporting
# ---------------------------------------------------------------------------

def _metric_cell(domain_report: Optional[Dict[str, Any]]) -> str:
    if not domain_report:
        return "N/A"
    mtype = str(domain_report.get("metric_type", "metric"))
    if mtype == "entity_f1":
        f1 = domain_report.get("overall", {}).get("f1", None)
        if f1 is None:
            return "N/A"
        return f"{float(f1):.4f}|entity_f1"
    if mtype == "accuracy":
        acc = domain_report.get("accuracy", None)
        if acc is None:
            return "N/A"
        return f"{float(acc):.4f}|accuracy"
    return "N/A"


def _write_transfer_matrix_csv(
    out_path: str,
    baseline_report: Dict[str, Any],
    steered_report: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["variant", "funsd_source", "rvl_cdip"]

    rows = [
        {
            "variant": "baseline",
            "funsd_source": _metric_cell(baseline_report.get("funsd_source")),
            "rvl_cdip": _metric_cell(baseline_report.get("rvl_cdip")),
        },
        {
            "variant": "concept_steered",
            "funsd_source": _metric_cell(steered_report.get("funsd_source")),
            "rvl_cdip": _metric_cell(steered_report.get("rvl_cdip")),
        },
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _write_entity_invariance_report(
    out_path: str,
    baseline_report: Dict[str, Any],
    steered_report: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_pc = baseline_report.get("funsd_source", {}).get("per_class", {}) or {}
    steered_pc = steered_report.get("funsd_source", {}).get("per_class", {}) or {}
    labels = sorted(set(base_pc.keys()) | set(steered_pc.keys()))

    lines: List[str] = [
        "# Entity Invariance Report",
        "",
        "This report summarizes FUNSD per-class F1 before and after concept steering.",
        "",
        "## Source (FUNSD) Per-Class F1",
    ]

    if not labels:
        lines.extend(
            [
                "",
                "- Per-class metrics are unavailable in baseline/steered reports.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "| Entity | Baseline F1 | Steered F1 | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label in labels:
            b_f1 = base_pc.get(label, {}).get("f1", None)
            s_f1 = steered_pc.get(label, {}).get("f1", None)
            if b_f1 is None and s_f1 is None:
                continue

            b_val = float(b_f1) if b_f1 is not None else 0.0
            s_val = float(s_f1) if s_f1 is not None else 0.0
            delta = s_val - b_val
            lines.append(f"| {label} | {b_val:.4f} | {s_val:.4f} | {delta:+.4f} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Subcommand operations
# ---------------------------------------------------------------------------

def _prepare_runtime(args):
    cache_dir = _resolve_cache_dir(args.cache_dir)
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    checkpoint = _resolve_path(args.checkpoint, DEFAULT_CHECKPOINT)
    device = _pick_device(args.device)

    os.makedirs(output_dir, exist_ok=True)
    return cache_dir, output_dir, checkpoint, device


def _run_prepare_data(args) -> Dict[str, Any]:
    cache_dir, output_dir, _, _ = _prepare_runtime(args)

    funsd_ds, funsd_status = _load_funsd_dataset(cache_dir)
    if funsd_ds is not None:
        funsd_status["splits"] = list(funsd_ds.keys())
        if "train" in funsd_ds:
            funsd_status["train_size"] = int(len(funsd_ds["train"]))
        if "test" in funsd_ds:
            funsd_status["test_size"] = int(len(funsd_ds["test"]))

    rvl_ds, rvl_status = _load_rvl_dataset(cache_dir)
    if rvl_ds is not None:
        split_name = args.rvl_split if args.rvl_split in rvl_ds else list(rvl_ds.keys())[0]
        rvl_status["split_selected"] = split_name
        rvl_status["split_size"] = int(len(rvl_ds[split_name]))



    errors: List[str] = []
    if funsd_ds is None:
        errors.append(str(funsd_status.get("reason", "FUNSD unavailable")))
    if rvl_ds is None:
        errors.append(str(rvl_status.get("reason", "RVL-CDIP unavailable")))

    payload = {
        "status": "ok" if (funsd_ds is not None and rvl_ds is not None) else "error",
        "reason": None if not errors else "; ".join(errors),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "download_mode": args.download_mode,
        "funsd_source": funsd_status,
        "rvl_cdip": rvl_status,
    }
    payload = _attach_provenance(payload, args, artifact_name="data_status")

    _json_dump(os.path.join(output_dir, "data_status.json"), payload)
    return payload


def _run_extract_concepts(args) -> Dict[str, Any]:
    cache_dir, output_dir, checkpoint, device = _prepare_runtime(args)

    model, processor, input_size = _load_model_and_processor(checkpoint, cache_dir, device)
    backend = OCRBackend(engine="doctr")

    dataset, funsd_status = _load_funsd_dataset(cache_dir)
    if dataset is None:
        raise RuntimeError(
            f"{funsd_status.get('reason', 'FUNSD unavailable')} "
            f"Action: {funsd_status.get('action', 'Please make FUNSD available and retry.')}"
        )
    vectors, stats = _extract_concept_vectors(
        model=model,
        processor=processor,
        dataset=dataset,
        input_size=input_size,
        ocr_backend=backend,
        limit=args.limit,
    )

    vec_path = _resolve_artifact_path(output_dir, args.concept_vector_path, "concept_vectors.pt")
    stats_path = _resolve_artifact_path(output_dir, args.concept_stats_path, "concept_vector_stats.json")

    _save_concept_vectors(vec_path, vectors)
    payload = {
        "status": "ok",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": checkpoint,
        "input_size": input_size,
        "concept_vector_path": vec_path,
        "concept_stats": stats,
    }
    payload = _attach_provenance(payload, args, artifact_name="concept_vector_stats")
    _json_dump(stats_path, payload)
    return payload


def _evaluate_all_domains(
    args,
    concept_vectors: Optional[Dict[str, torch.Tensor]],
    alpha: float,
) -> Dict[str, Any]:
    cache_dir, output_dir, checkpoint, device = _prepare_runtime(args)

    model, processor, input_size = _load_model_and_processor(checkpoint, cache_dir, device)
    backend = OCRBackend(engine="doctr")

    funsd_ds, funsd_status = _load_funsd_dataset(cache_dir)
    if funsd_ds is None:
        raise RuntimeError(
            f"{funsd_status.get('reason', 'FUNSD unavailable')} "
            f"Action: {funsd_status.get('action', 'Please make FUNSD available and retry.')}"
        )
    label_list = funsd_ds["train"].features["ner_tags"].feature.names
    label2id = {v: i for i, v in enumerate(label_list)}
    o_label_id = int(label2id.get("O", 0))

    print("[eval] preparing FUNSD source split ...")
    funsd_samples, funsd_stats = _prepare_funsd_samples(
        split=funsd_ds["test"],
        processor=processor,
        ocr_backend=backend,
        o_label_id=o_label_id,
        input_size=input_size,
        limit=args.limit,
    )

    print("[eval] FUNSD entity evaluation ...")
    funsd_report = _evaluate_funsd_entity(
        model=model,
        samples=funsd_samples,
        label_list=label_list,
        cache_dir=cache_dir,
        concept_vectors=concept_vectors,
        alpha=alpha,
    )

    print("[eval] RVL-CDIP setup ...")
    rvl_ds, rvl_status = _load_rvl_dataset(cache_dir)
    if rvl_ds is None:
        rvl_report = {
            "status": "skipped_with_reason",
            "metric_type": "accuracy",
            "reason": rvl_status,
        }
    else:
        split_name = args.rvl_split if args.rvl_split in rvl_ds else list(rvl_ds.keys())[0]
        if "train" not in rvl_ds:
            rvl_report = {
                "status": "skipped_with_reason",
                "metric_type": "accuracy",
                "reason": {
                    "message": (
                        "RVL train split is unavailable. Prototype construction from eval split "
                        "is disabled to avoid data leakage."
                    ),
                    "available_splits": list(rvl_ds.keys()),
                    "requested_eval_split": split_name,
                },
            }
        else:
            proto_split = rvl_ds["train"]
            print("[eval] RVL prototype build (no fine-tune) ...")
            prototypes = _build_rvl_prototypes(
                model=model,
                processor=processor,
                dataset_split=proto_split,
                input_size=input_size,
                ocr_backend=backend,
                train_limit=args.rvl_train_limit,
            )
            print("[eval] RVL evaluation ...")
            rvl_report = _evaluate_rvl(
                model=model,
                processor=processor,
                dataset_split=rvl_ds[split_name],
                input_size=input_size,
                ocr_backend=backend,
                prototypes=prototypes,
                concept_vectors=concept_vectors,
                alpha=alpha,
                limit=args.limit,
            )
            rvl_report["split"] = split_name

    print("[eval] CORD setup (OOD) ...")
    cord_ds, cord_status = _load_cord_dataset(cache_dir)
    if cord_ds is None:
        cord_report = {"status": "skipped", "reason": cord_status}
    else:
        cord_split = args.cord_split if args.cord_split in cord_ds else list(cord_ds.keys())[0]
        if "train" not in cord_ds:
            cord_report = {"status": "skipped", "reason": "No train split in CORD for prototypes"}
        else:
            print("[eval] CORD prototype build ...")
            c_protos = _build_cord_prototypes(
                model=model,
                processor=processor,
                dataset_split=cord_ds["train"],
                input_size=input_size,
                ocr_backend=backend,
                train_limit=args.cord_train_limit,
            )
            print("[eval] CORD evaluation ...")
            cord_report = _evaluate_cord(
                model=model,
                processor=processor,
                dataset_split=cord_ds[cord_split],
                input_size=input_size,
                ocr_backend=backend,
                prototypes=c_protos,
                concept_vectors=concept_vectors,
                alpha=alpha,
                limit=args.limit,
            )

    return {
        "status": "ok",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": checkpoint,
        "alpha": float(alpha),
        "funsd_source": funsd_report,
        "rvl_cdip": rvl_report,
        "cord_v2": cord_report,
        "prep_stats": {
            "funsd": funsd_stats,
            "funsd_status": funsd_status,
            "rvl_status": rvl_status,
        },
    }


def _run_eval_baseline(args) -> Dict[str, Any]:
    payload = _evaluate_all_domains(args=args, concept_vectors=None, alpha=0.0)
    payload["mode"] = "baseline"
    payload = _attach_provenance(payload, args, artifact_name="baseline_eval")
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    _json_dump(os.path.join(output_dir, "baseline_eval.json"), payload)
    return payload


def _run_eval_steered(args) -> Dict[str, Any]:
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    vector_path = _resolve_artifact_path(output_dir, args.concept_vector_path, "concept_vectors.pt")
    if not os.path.exists(vector_path):
        raise FileNotFoundError(
            f"Concept vector file not found: {vector_path}. "
            "Run extract_concepts first."
        )
    concept_vectors = _load_concept_vectors(vector_path)

    alphas = list(args.alpha_sweep) if args.alpha_sweep else [float(args.alpha)]
    sweeps = []
    for a in alphas:
        rep = _evaluate_all_domains(args=args, concept_vectors=concept_vectors, alpha=float(a))
        rep["mode"] = "steered"
        sweeps.append(rep)

    # Pick alpha with highest available target score priority: RVL accuracy, then FUNSD f1.
    def _score(rep):
        rvl = rep.get("rvl_cdip", {})
        if "accuracy" in rvl:
            return float(rvl.get("accuracy", 0.0))
        return float(rep.get("funsd_source", {}).get("overall", {}).get("f1", 0.0))

    best = max(sweeps, key=_score)
    best["alpha_sweep"] = [{"alpha": s["alpha"], "score": _score(s)} for s in sweeps]
    best = _attach_provenance(best, args, artifact_name="steered_eval")

    _json_dump(os.path.join(output_dir, "steered_eval.json"), best)
    return best


def _run_transfer_matrix(args) -> Dict[str, Any]:
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    baseline_path = os.path.join(output_dir, "baseline_eval.json")
    steered_path = os.path.join(output_dir, "steered_eval.json")

    if not os.path.exists(baseline_path) or not os.path.exists(steered_path):
        raise FileNotFoundError(
            "baseline_eval.json or steered_eval.json not found. "
            "Run eval_baseline and eval_steered first."
        )

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_report = json.load(f)
    with open(steered_path, "r", encoding="utf-8") as f:
        steered_report = json.load(f)

    matrix_path = _resolve_artifact_path(output_dir, args.matrix_path, "transfer_matrix.csv")
    _write_transfer_matrix_csv(matrix_path, baseline_report, steered_report)

    invariance_path = _resolve_artifact_path(output_dir, args.invariance_report_path, "entity_invariance_report.md")
    _write_entity_invariance_report(
        out_path=invariance_path,
        baseline_report=baseline_report,
        steered_report=steered_report,
    )

    payload = {
        "status": "ok",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "matrix_path": matrix_path,
        "invariance_report_path": invariance_path,
        "baseline": baseline_report,
        "steered": steered_report,
    }
    payload = _attach_provenance(payload, args, artifact_name="domain_transfer_report")

    final_report_path = _resolve_artifact_path(output_dir, args.report_path, "domain_transfer_report.json")
    _json_dump(final_report_path, payload)
    return payload


def _run_all(args) -> Dict[str, Any]:
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    data_status = _run_prepare_data(args)
    if data_status.get("status") != "ok":
        return {
            "status": "error",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "reason": data_status.get("reason", "Required datasets are unavailable."),
            "data_status": data_status,
            "action": "Run prepare_data, resolve dataset access/download issues, then retry run_all.",
        }

    vector_path = _resolve_artifact_path(output_dir, args.concept_vector_path, "concept_vectors.pt")
    if not os.path.exists(vector_path):
        print("[run_all] concept vectors missing; extracting ...")
        _run_extract_concepts(args)

    baseline = _run_eval_baseline(args)
    steered = _run_eval_steered(args)
    matrix = _run_transfer_matrix(args)

    payload = {
        "status": "ok",
        "mode": "run_all",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "data_status": data_status,
        "baseline": baseline,
        "steered": steered,
        "transfer_matrix": {
            "matrix_path": matrix.get("matrix_path"),
            "invariance_report_path": matrix.get("invariance_report_path"),
        },
    }
    payload = _attach_provenance(payload, args, artifact_name="domain_transfer_report")
    final_report_path = _resolve_artifact_path(output_dir, args.report_path, "domain_transfer_report.json")
    _json_dump(final_report_path, payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="HF cache directory")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Artifacts output directory")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Source FUNSD checkpoint")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Device")
    p.add_argument("--limit", type=int, default=None, help="Optional document cap for smoke tests")
    p.add_argument("--rvl_split", type=str, default="test", help="RVL split for evaluation")
    p.add_argument("--rvl_train_limit", type=int, default=1500, help="RVL train docs used for prototype build")
    p.add_argument("--cord_split", type=str, default="test", help="CORD split for evaluation")
    p.add_argument("--cord_train_limit", type=int, default=500, help="CORD train docs used for prototype build")
    p.add_argument("--download_mode", type=str, default="auto", choices=["auto", "manual"], help="Dataset acquisition mode")
    p.add_argument("--concept_vector_path", type=str, default=None, help="Concept vector .pt path (default: <output_dir>/concept_vectors.pt)")
    p.add_argument("--concept_stats_path", type=str, default=None, help="Concept stats JSON path (default: <output_dir>/concept_vector_stats.json)")
    p.add_argument("--matrix_path", type=str, default=None, help="Transfer matrix CSV path (default: <output_dir>/transfer_matrix.csv)")
    p.add_argument("--invariance_report_path", type=str, default=None, help="Entity invariance markdown path (default: <output_dir>/entity_invariance_report.md)")
    p.add_argument("--report_path", type=str, default=None, help="Final report JSON path (default: <output_dir>/domain_transfer_report.json)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OOD generalization + test-time concept steering pipeline (LayoutLMv3, no fine-tuning)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare_data", help="Download/validate RVL and FUNSD datasets")
    _add_common_args(p_prepare)

    p_extract = sub.add_parser("extract_concepts", help="Extract 4 concept vectors from source-domain hidden states")
    _add_common_args(p_extract)

    p_base = sub.add_parser("eval_baseline", help="Evaluate OOD generalization without steering")
    _add_common_args(p_base)

    p_steer = sub.add_parser("eval_steered", help="Evaluate OOD generalization with concept steering")
    _add_common_args(p_steer)
    p_steer.add_argument("--alpha", type=float, default=0.15, help="Steering strength")
    p_steer.add_argument("--alpha_sweep", nargs="*", type=float, default=None, help="Optional alpha sweep")

    p_matrix = sub.add_parser("transfer_matrix", help="Build domain transfer matrix and invariance report")
    _add_common_args(p_matrix)

    p_all = sub.add_parser("run_all", help="Orchestrate full Task-5 flow")
    _add_common_args(p_all)
    p_all.add_argument("--alpha", type=float, default=0.15, help="Steering strength")
    p_all.add_argument("--alpha_sweep", nargs="*", type=float, default=None, help="Optional alpha sweep")

    return parser


def _print_summary(payload: Dict[str, Any]) -> None:
    mode = payload.get("mode", payload.get("command", "result"))
    print(f"\n=== {str(mode).upper()} SUMMARY ===")

    if payload.get("status") != "ok":
        print(payload.get("error", payload.get("reason", "Unknown error")))
        if "rvl_cdip" in payload:
            rvl = payload.get("rvl_cdip", {})
            if isinstance(rvl, dict):
                print(f"RVL status: {rvl.get('status', 'unknown')}")
        return

    if "baseline" in payload and "steered" in payload:
        base = payload["baseline"]
        steer = payload["steered"]
        b_funsd = base.get("funsd_source", {}).get("overall", {}).get("f1", "n/a")
        s_funsd = steer.get("funsd_source", {}).get("overall", {}).get("f1", "n/a")
        b_rvl = base.get("rvl_cdip", {}).get("accuracy", "n/a")
        s_rvl = steer.get("rvl_cdip", {}).get("accuracy", "n/a")
        b_cord = base.get("cord_v2", {}).get("accuracy", "n/a")
        s_cord = steer.get("cord_v2", {}).get("accuracy", "n/a")

        b_funsd_token = base.get("funsd_source", {}).get("token_f1", "n/a")
        s_funsd_token = steer.get("funsd_source", {}).get("token_f1", "n/a")

        print(f"FUNSD source entity_f1: baseline={b_funsd} steered={s_funsd}")
        print(f"FUNSD source token_f1:  baseline={b_funsd_token} steered={s_funsd_token}")
        print(f"RVL accuracy: baseline={b_rvl} steered={s_rvl}")
        print(f"CORD accuracy: baseline={b_cord} steered={s_cord}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "prepare_data":
            payload = _run_prepare_data(args)
            payload["mode"] = "prepare_data"
        elif args.command == "extract_concepts":
            payload = _run_extract_concepts(args)
            payload["mode"] = "extract_concepts"
        elif args.command == "eval_baseline":
            payload = _run_eval_baseline(args)
            payload["mode"] = "eval_baseline"
        elif args.command == "eval_steered":
            payload = _run_eval_steered(args)
            payload["mode"] = "eval_steered"
        elif args.command == "transfer_matrix":
            payload = _run_transfer_matrix(args)
            payload["mode"] = "transfer_matrix"
        elif args.command == "run_all":
            payload = _run_all(args)
            payload["mode"] = "run_all"
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as e:
        payload = {
            "status": "error",
            "mode": args.command,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
        }

    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    report_path = _resolve_artifact_path(output_dir, args.report_path, "domain_transfer_report.json")
    # Always emit a command-level report snapshot at the reported path.
    payload = _attach_provenance(payload, args, artifact_name="domain_transfer_report")
    _json_dump(report_path, payload)

    print(json.dumps({"status": payload.get("status", "unknown"), "report_path": report_path}, indent=2))
    _print_summary(payload)


if __name__ == "__main__":
    main()
