"""
multi_resolution_pipeline.py
============================
CLI-only multi-resolution inference pipeline for FUNSD + LayoutLMv3 + docTR.

Features:
  1) Optional train-missing flow for 224/384/512 checkpoints.
  2) Fixed-resolution benchmark (F1 + end-to-end latency).
  3) Adaptive complexity selector (element_count + text_density).
  4) Cascaded inference (224 -> 384 when confidence < threshold).
"""

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import evaluate as hf_evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from layoutlm_customOCR import OCRBackend, align_ocr_to_ground_truth, train as train_layoutlm

DEFAULT_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, ".hf_cache")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "eval_results")
DEFAULT_RESOLUTIONS = [224, 384, 512]


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


def _resolve_cache_dir(cache_dir: str) -> str:
    # Guard against placeholder examples accidentally copied from docs/CLI help.
    invalid_prefixes = ("/absolute", "<absolute", "ABSOLUTE_PATH")
    if any(str(cache_dir).startswith(p) for p in invalid_prefixes):
        fallback = DEFAULT_CACHE_DIR
        print(
            f"[setup] WARNING: cache_dir='{cache_dir}' looks like a placeholder. "
            f"Falling back to '{fallback}'."
        )
        cache_dir = fallback

    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Unable to create/use cache_dir '{cache_dir}'. "
            "Pass a writable path (for example './.hf_cache' or '$HOME/.cache/huggingface')."
        ) from e
    return cache_dir


def _find_best_checkpoint_with_metric(model_dir: str):
    best_metric = float("-inf")
    best_ckpt = None

    for sub in os.listdir(model_dir):
        if not sub.startswith("checkpoint-"):
            continue
        state_path = os.path.join(model_dir, sub, "trainer_state.json")
        if not os.path.exists(state_path):
            continue
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            metric = state.get("best_metric", float("-inf"))
            metric = float(metric) if metric is not None else float("-inf")
            if metric > best_metric:
                best_metric = metric
                best_ckpt = os.path.join(model_dir, sub)
        except Exception:
            continue

    if best_ckpt is None:
        ckpts = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d))
        ]
        if ckpts:
            best_ckpt = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1]))
            best_metric = float("-inf")

    return best_ckpt, best_metric


def _read_input_size(checkpoint_dir: str):
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        size = cfg.get("input_size", None)
        if size is None:
            return None
        return int(size)
    except Exception:
        return None


def _discover_resolution_models(layoutlm_dir: str, resolutions):
    wanted = set(int(r) for r in resolutions)
    chosen = {}

    for name in sorted(os.listdir(layoutlm_dir)):
        if not name.startswith("layoutlmv3-funsd-doctr"):
            continue
        model_dir = os.path.join(layoutlm_dir, name)
        if not os.path.isdir(model_dir):
            continue

        checkpoint, best_metric = _find_best_checkpoint_with_metric(model_dir)
        if checkpoint is None:
            continue

        input_size = _read_input_size(checkpoint)
        if input_size not in wanted:
            continue

        score = best_metric if best_metric != float("-inf") else -1e9
        prev = chosen.get(input_size)
        if prev is None or score > prev["score"]:
            chosen[input_size] = {
                "resolution": input_size,
                "model_dir": name,
                "checkpoint": checkpoint,
                "best_metric": None if best_metric == float("-inf") else float(best_metric),
                "score": score,
            }

    return {k: {kk: vv for kk, vv in v.items() if kk != "score"} for k, v in chosen.items()}


@contextmanager
def _pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _ensure_models(layoutlm_dir: str, cache_dir: str, resolutions, train_missing: bool):
    discovered = _discover_resolution_models(layoutlm_dir, resolutions)
    missing = [r for r in sorted(resolutions) if r not in discovered]
    if not missing:
        return discovered

    if not train_missing:
        raise FileNotFoundError(
            f"Missing checkpoints for resolutions: {missing}. "
            "Re-run with --train_missing to train only the missing ones."
        )

    for res in missing:
        print(f"[train_missing] Training missing resolution {res} ...")
        with _pushd(layoutlm_dir):
            train_layoutlm(
                ocr_engine="doctr",
                arch="layoutlmv3",
                model_size="base",
                cache_dir=cache_dir,
                target_size=int(res),
                output_suffix=f"res{res}",
            )

    discovered = _discover_resolution_models(layoutlm_dir, resolutions)
    still_missing = [r for r in sorted(resolutions) if r not in discovered]
    if still_missing:
        raise RuntimeError(
            f"Training completed but checkpoints were not discovered for: {still_missing}"
        )
    return discovered


def _pad_to_square(image: Image.Image, target: int) -> Image.Image:
    w, h = image.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    canvas.paste(image.resize((nw, nh), Image.LANCZOS), ((target - nw) // 2, (target - nh) // 2))
    return canvas


def _box_area_1000(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _compute_complexity_calibration(train_split):
    counts = []
    densities = []
    scores = []

    for ex in train_split:
        cnt = len(ex["words"])
        density = sum(_box_area_1000(b) for b in ex["bboxes"]) / 1_000_000.0
        counts.append(cnt)
        densities.append(density)

    p90_count = float(np.percentile(counts, 90)) if counts else 1.0
    p90_density = float(np.percentile(densities, 90)) if densities else 1.0
    p90_count = max(p90_count, 1e-6)
    p90_density = max(p90_density, 1e-6)

    for cnt, density in zip(counts, densities):
        count_norm = float(np.clip(cnt / p90_count, 0.0, 1.0))
        density_norm = float(np.clip(density / p90_density, 0.0, 1.0))
        score = 0.5 * count_norm + 0.5 * density_norm
        scores.append(score)

    q33 = float(np.quantile(scores, 0.33)) if scores else 0.33
    q66 = float(np.quantile(scores, 0.66)) if scores else 0.66

    return {
        "p90_count": p90_count,
        "p90_density": p90_density,
        "q33": q33,
        "q66": q66,
    }


def _complexity_from_ocr(words, boxes_1000, calib):
    element_count = len(words)
    text_density = sum(_box_area_1000(b) for b in boxes_1000) / 1_000_000.0
    count_norm = float(np.clip(element_count / calib["p90_count"], 0.0, 1.0))
    density_norm = float(np.clip(text_density / calib["p90_density"], 0.0, 1.0))
    score = 0.5 * count_norm + 0.5 * density_norm
    return {
        "element_count": element_count,
        "text_density": text_density,
        "count_norm": count_norm,
        "density_norm": density_norm,
        "score": score,
    }


def _select_resolution(score: float, calib, sorted_resolutions):
    low_res = sorted_resolutions[0]
    high_res = sorted_resolutions[-1]
    mid_res = sorted_resolutions[len(sorted_resolutions) // 2]

    if score <= calib["q33"]:
        return low_res
    if score >= calib["q66"]:
        return high_res
    return mid_res


def _load_models(checkpoint_map, cache_dir: str, device: torch.device):
    loaded = {}
    for resolution in sorted(checkpoint_map.keys()):
        info = checkpoint_map[resolution]
        checkpoint = info["checkpoint"]
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
            cache_dir=cache_dir,
        )
        processor.image_processor.size = {"height": resolution, "width": resolution}
        processor.image_processor.do_resize = True
        processor.image_processor.do_pad = True

        model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint).to(device)
        model.eval()
        if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "init_visual_bbox"):
            try:
                grid = resolution // 16
                model.layoutlmv3.init_visual_bbox(image_size=(grid, grid))
            except Exception:
                pass

        id2label = {int(k): v for k, v in model.config.id2label.items()}
        label_list = [id2label[i] for i in range(len(id2label))]
        label2id = {v: k for k, v in id2label.items()}

        loaded[resolution] = {
            "model": model,
            "processor": processor,
            "id2label": id2label,
            "label_list": label_list,
            "label2id": label2id,
        }
    return loaded


def _infer_single(model_pack, image, words, boxes_1000, labels, device, resolution):
    t0 = time.perf_counter()
    image_sq = _pad_to_square(image, resolution)
    encoding = model_pack["processor"](
        image_sq,
        words,
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

    with torch.no_grad():
        logits = model_pack["model"](**inputs).logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(-1).cpu().tolist()
    max_probs = probs.max(dim=-1).values.cpu().tolist()
    model_ms = (time.perf_counter() - t0) * 1000.0

    prev_word_id = None
    doc_true = []
    doc_pred = []
    word_conf = []
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev_word_id:
            true_lid = input_label_ids[pos]
            if true_lid != -100:
                doc_true.append(model_pack["label_list"][true_lid])
                doc_pred.append(model_pack["label_list"][pred_ids[pos]])
                word_conf.append(float(max_probs[pos]))
            prev_word_id = wid

    confidence = float(np.mean(word_conf)) if word_conf else 0.0
    return {
        "doc_true": doc_true,
        "doc_pred": doc_pred,
        "confidence": confidence,
        "model_ms": model_ms,
    }


def _seqeval_summary(seqeval_metric, true_all, pred_all):
    result = seqeval_metric.compute(predictions=pred_all, references=true_all)
    return {
        "precision": round(float(result["overall_precision"]), 4),
        "recall": round(float(result["overall_recall"]), 4),
        "f1": round(float(result["overall_f1"]), 4),
        "accuracy": round(float(result["overall_accuracy"]), 4),
    }


def _evaluate(dataset, loaded_models, calib, confidence_threshold: float, limit: int = None):
    sorted_res = sorted(loaded_models.keys())
    low_res = sorted_res[0]
    mid_res = sorted_res[len(sorted_res) // 2]
    high_res = sorted_res[-1]

    seqeval_metric = hf_evaluate.load("seqeval")
    ocr_backend = OCRBackend(engine="doctr")
    o_label_id = loaded_models[low_res]["label2id"].get("O", 0)

    test_split = dataset["test"]
    if limit is not None:
        test_split = test_split.select(range(min(limit, len(test_split))))

    fixed = {
        r: {"true": [], "pred": [], "lat_ms": []}
        for r in sorted_res
    }
    adaptive = {"true": [], "pred": [], "lat_ms": [], "resolution_counts": {str(r): 0 for r in sorted_res}}
    cascade = {"true": [], "pred": [], "lat_ms": [], "escalations": 0, "total_docs": 0}

    for idx, ex in enumerate(test_split):
        print(f"[eval] {idx + 1}/{len(test_split)}", end="\r")

        image = ex["image"].convert("RGB")
        gt_words = ex["words"]
        gt_boxes = ex["bboxes"]
        gt_labels = ex["ner_tags"]

        t_ocr0 = time.perf_counter()
        words, boxes_1000, _, _ = ocr_backend.run(image)
        ocr_ms = (time.perf_counter() - t_ocr0) * 1000.0

        if not words:
            words = [""]
            boxes_1000 = [[0, 0, 0, 0]]

        t_align0 = time.perf_counter()
        aligned_labels = align_ocr_to_ground_truth(
            words,
            boxes_1000,
            gt_words,
            gt_boxes,
            gt_labels,
            default_label_id=o_label_id,
        )
        align_ms = (time.perf_counter() - t_align0) * 1000.0

        complexity = _complexity_from_ocr(words, boxes_1000, calib)
        selected_res = _select_resolution(complexity["score"], calib, sorted_res)

        outputs_by_res = {}
        for res in sorted_res:
            out = _infer_single(
                model_pack=loaded_models[res],
                image=image,
                words=words,
                boxes_1000=boxes_1000,
                labels=aligned_labels,
                device=next(loaded_models[res]["model"].parameters()).device,
                resolution=res,
            )
            end_to_end_ms = max(0.0, ocr_ms + out["model_ms"])
            fixed[res]["true"].append(out["doc_true"])
            fixed[res]["pred"].append(out["doc_pred"])
            fixed[res]["lat_ms"].append(end_to_end_ms)
            outputs_by_res[res] = out

        adaptive_out = outputs_by_res[selected_res]
        adaptive["true"].append(adaptive_out["doc_true"])
        adaptive["pred"].append(adaptive_out["doc_pred"])
        adaptive["lat_ms"].append(max(0.0, ocr_ms + adaptive_out["model_ms"]))
        adaptive["resolution_counts"][str(selected_res)] += 1

        cascade["total_docs"] += 1
        low_out = outputs_by_res[low_res]
        if low_out["confidence"] < confidence_threshold:
            cascade_out = outputs_by_res[mid_res]
            cascade["escalations"] += 1
            cascade_ms = max(0.0, ocr_ms + low_out["model_ms"] + cascade_out["model_ms"])
        else:
            cascade_out = low_out
            cascade_ms = max(0.0, ocr_ms + low_out["model_ms"])

        cascade["true"].append(cascade_out["doc_true"])
        cascade["pred"].append(cascade_out["doc_pred"])
        cascade["lat_ms"].append(cascade_ms)

        _ = align_ms  # alignment is for evaluation labels only, excluded from latency.

    print()

    fixed_report = {}
    for res in sorted_res:
        fixed_report[str(res)] = {
            "metrics": _seqeval_summary(seqeval_metric, fixed[res]["true"], fixed[res]["pred"]),
            "avg_latency_ms": round(float(statistics.mean(fixed[res]["lat_ms"])) if fixed[res]["lat_ms"] else 0.0, 3),
            "docs": len(fixed[res]["lat_ms"]),
        }

    adaptive_report = {
        "metrics": _seqeval_summary(seqeval_metric, adaptive["true"], adaptive["pred"]),
        "avg_latency_ms": round(float(statistics.mean(adaptive["lat_ms"])) if adaptive["lat_ms"] else 0.0, 3),
        "resolution_counts": adaptive["resolution_counts"],
        "docs": len(adaptive["lat_ms"]),
    }

    cascade_report = {
        "metrics": _seqeval_summary(seqeval_metric, cascade["true"], cascade["pred"]),
        "avg_latency_ms": round(float(statistics.mean(cascade["lat_ms"])) if cascade["lat_ms"] else 0.0, 3),
        "escalation_rate": round(
            float(cascade["escalations"] / max(cascade["total_docs"], 1)),
            4,
        ),
        "threshold": confidence_threshold,
        "docs": len(cascade["lat_ms"]),
    }

    if str(high_res) in fixed_report and fixed_report[str(high_res)]["avg_latency_ms"] > 0:
        base = fixed_report[str(high_res)]["avg_latency_ms"]
        adaptive_speedup = 100.0 * (base - adaptive_report["avg_latency_ms"]) / base
        cascade_speedup = 100.0 * (base - cascade_report["avg_latency_ms"]) / base
        adaptive_report["speedup_vs_high_res_pct"] = round(adaptive_speedup, 2)
        cascade_report["speedup_vs_high_res_pct"] = round(cascade_speedup, 2)

    return fixed_report, adaptive_report, cascade_report


def _print_summary(fixed_report, adaptive_report, cascade_report):
    print("\n=== Fixed Resolution Benchmark ===")
    print("resolution | f1     | latency_ms")
    print("----------------------------------")
    for res, payload in sorted(fixed_report.items(), key=lambda x: int(x[0])):
        print(
            f"{res:>10} | {payload['metrics']['f1']:.4f} | {payload['avg_latency_ms']:.2f}"
        )

    print("\n=== Adaptive Selector ===")
    print(
        f"f1={adaptive_report['metrics']['f1']:.4f} "
        f"latency_ms={adaptive_report['avg_latency_ms']:.2f} "
        f"counts={adaptive_report['resolution_counts']}"
    )

    print("\n=== Cascade 224->384 ===")
    print(
        f"f1={cascade_report['metrics']['f1']:.4f} "
        f"latency_ms={cascade_report['avg_latency_ms']:.2f} "
        f"escalation_rate={cascade_report['escalation_rate']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-resolution LayoutLMv3 pipeline (FUNSD + docTR).")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for final JSON report")
    parser.add_argument("--train_missing", action="store_true", help="Train only missing resolution checkpoints")
    parser.add_argument("--resolutions", nargs="+", type=int, default=DEFAULT_RESOLUTIONS, help="Resolutions to use (default: 224 384 512)")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Cascade threshold for 224->384 escalation")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on FUNSD test docs for smoke runs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Inference device")
    args = parser.parse_args()

    resolutions = sorted(set(int(r) for r in args.resolutions))
    if len(resolutions) < 3:
        raise ValueError("Please provide at least 3 resolutions (e.g. 224 384 512).")

    os.makedirs(args.output_dir, exist_ok=True)
    device = _pick_device(args.device)
    cache_dir = _resolve_cache_dir(args.cache_dir)

    print("[setup] Loading FUNSD dataset ...")
    dataset = load_dataset("nielsr/funsd", cache_dir=cache_dir)
    print(f"[setup] Device: {device}")
    print(f"[setup] Resolutions: {resolutions}")
    print(f"[setup] Cache dir: {cache_dir}")

    print("[setup] Discovering / preparing checkpoints ...")
    checkpoint_map = _ensure_models(
        layoutlm_dir=SCRIPT_DIR,
        cache_dir=cache_dir,
        resolutions=resolutions,
        train_missing=args.train_missing,
    )
    for res in resolutions:
        info = checkpoint_map[res]
        rel_ckpt = os.path.relpath(info["checkpoint"], SCRIPT_DIR)
        print(f"  - {res}: {info['model_dir']} ({rel_ckpt})")

    print("[setup] Loading models ...")
    loaded_models = _load_models(checkpoint_map, cache_dir=cache_dir, device=device)

    print("[setup] Computing complexity calibration (train split GT stats) ...")
    calibration = _compute_complexity_calibration(dataset["train"])

    print("[run] Evaluating fixed/adaptive/cascade flows ...")
    fixed_report, adaptive_report, cascade_report = _evaluate(
        dataset=dataset,
        loaded_models=loaded_models,
        calib=calibration,
        confidence_threshold=args.confidence_threshold,
        limit=args.limit,
    )

    _print_summary(fixed_report, adaptive_report, cascade_report)

    report = {
        "metadata": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "device": str(device),
            "cache_dir": cache_dir,
            "resolutions": resolutions,
            "confidence_threshold": args.confidence_threshold,
            "limit": args.limit,
            "dataset": "nielsr/funsd",
            "split_for_eval": "test",
            "ocr_engine": "doctr",
            "latency_scope": "end_to_end_excluding_model_load",
        },
        "checkpoints": {
            str(r): {
                "model_dir": checkpoint_map[r]["model_dir"],
                "checkpoint": os.path.relpath(checkpoint_map[r]["checkpoint"], SCRIPT_DIR),
                "best_metric": checkpoint_map[r]["best_metric"],
            }
            for r in resolutions
        },
        "complexity_calibration": calibration,
        "fixed_resolution_benchmark": fixed_report,
        "adaptive_eval": adaptive_report,
        "cascade_eval": cascade_report,
    }

    out_path = os.path.join(args.output_dir, "multi_resolution_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {out_path}")


if __name__ == "__main__":
    main()
