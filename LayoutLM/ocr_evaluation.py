"""
ocr_evaluation.py
==================
Standalone OCR quality evaluation against FUNSD ground-truth annotations.

Metrics:
  1. Character-level: CER, WER  (via jiwer)
  2. Word Recall: exact/fuzzy match analysis
  3. Segmentation: over-seg, under-seg, reading-order consistency
  4. Spatial Accuracy: IoU statistics for matched words

Usage:
  python ocr_evaluation.py --ocr_engine tesseract --data_dir /path/to/FUNSD
  python ocr_evaluation.py --ocr_engine paddle --data_dir /path/to/FUNSD --split all
  python ocr_evaluation.py --compare --data_dir /path/to/FUNSD --split all
"""

import os
import sys
import json
import glob
import re
import argparse
import statistics
import time
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from jiwer import wer as compute_wer, cer as compute_cer

# Optional denoising (OpenCV if available; falls back to PIL median filter)
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ===========================================================================
# GPU-aware OCR Backend (standalone — does not modify model code)
# ===========================================================================

class FastOCRBackend:
    """
    GPU-accelerated OCR backend for evaluation.
    - PaddleOCR: uses GPU when CUDA is available, CPU otherwise.
    - Tesseract: CPU-only (no GPU path), but we enable parallel processing.
    """

    def __init__(self, engine="tesseract"):
        self.engine = engine

        if self.engine in ("paddle", "paddle-v4"):
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                raise RuntimeError("PaddleOCR not installed. pip install paddleocr")

            # Determine GPU usage for PaddleOCR
            if sys.platform == "darwin":
                use_gpu = False
            else:
                force_cpu = os.getenv("PADDLE_OCR_FORCE_CPU", "0") == "1"
                if force_cpu:
                    use_gpu = False
                else:
                    try:
                        import paddle
                        use_gpu = paddle.device.is_compiled_with_cuda()
                    except Exception:
                        use_gpu = False
            gpu_label = "GPU (CUDA)" if use_gpu else "CPU"
            # paddle-v4 uses PP-OCRv4 (highest accuracy in PaddleOCR 2.x)
            ocr_version = "PP-OCRv4" if self.engine == "paddle-v4" else "PP-OCRv3"
            print(f"  PaddleOCR ({ocr_version}) initialising on {gpu_label} ...")
            self.paddle = PaddleOCR(
                use_textline_orientation=True,
                lang="en",
                show_log=False,
                use_gpu=use_gpu,
                ocr_version=ocr_version,
            )

        elif self.engine == "paddle-vl":
            # PaddleOCR 3.x Vision-Language model (PP-ChatOCRv3 / multimodal).
            # Requires: pip install paddlepaddle paddleocr>=3.0
            try:
                from paddleocr import PaddleOCR as _PaddleOCR
                import paddle as _paddle
                ver = getattr(_paddle, '__version__', '0')
                major = int(str(ver).split('.')[0])
                # paddleocr 3.x exposes the VL pipeline differently
                # Try the new API; fall back with a clear error if not available.
                # New 3.x: PaddleOCR(use_vl=True) or specific pipeline names.
                print(f"  PaddleOCR VL initialising (paddle={ver}) ...")
                self.paddle = _PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                    show_log=False,
                    use_gpu=False,  # MPS not supported by Paddle
                    ocr_version="PP-OCRv4",  # closest in 2.x; 3.x adds VL pipeline
                )
                self._paddle_vl_native = False  # flag: running in compat mode
                if major >= 3:
                    # Try to import the 3.x VL predictor if available
                    try:
                        from paddleocr import DocVLMPredictor  # noqa: F401
                        self._paddle_vl_native = True
                        print("  ✓ Native PaddleOCR 3.x VL predictor available.")
                    except ImportError:
                        print("  ⚠ PaddleOCR 3.x installed but DocVLMPredictor not found; "
                              "using PP-OCRv4 fallback.")
                else:
                    print("  ⚠ paddle-vl requires PaddleOCR>=3.0 for full VL support. "
                          f"Current version: {ver}. Running PP-OCRv4 as fallback. "
                          "To upgrade: pip install paddlepaddle paddleocr>=3.0")
            except ImportError:
                raise RuntimeError(
                    "PaddleOCR not installed. To use paddle-vl:\n"
                    "  pip install paddlepaddle paddleocr>=3.0"
                )

        elif self.engine == "tesseract":
            try:
                import pytesseract
                if os.path.exists("/opt/homebrew/bin/tesseract"):
                    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
                self.pytesseract = pytesseract
            except ImportError:
                raise RuntimeError("pytesseract not installed. pip install pytesseract")

        elif self.engine == "doctr":
            try:
                import torch
                from doctr.models import ocr_predictor
            except ImportError:
                raise RuntimeError("python-doctr not installed. pip install \"python-doctr[torch]\"")

            self.doctr_device = torch.device(
                "mps"  if torch.backends.mps.is_available() else
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"  docTR device: {self.doctr_device}")
            self.doctr = ocr_predictor(
                pretrained=True,
                assume_straight_pages=True,
                export_as_straight_boxes=True,
            )
            self.doctr.to(self.doctr_device)

    def run(self, image: Image.Image):
        """
        Run OCR. Returns (words, boxes_1000, confidences, boxes_px).
        All boxes normalised to 0-1000 LayoutLM scale.
        """
        width, height = image.size
        words, boxes_1000, boxes_px, confidences = [], [], [], []

        if self.engine == "tesseract":
            data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if not text:
                    continue
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                conf = float(data["conf"][i])
                px_box = [float(x), float(y), float(x + w), float(y + h)]
                x1 = max(0, min(1000, int(1000 * x / width)))
                y1 = max(0, min(1000, int(1000 * y / height)))
                x2 = max(0, min(1000, int(1000 * (x + w) / width)))
                y2 = max(0, min(1000, int(1000 * (y + h) / height)))
                words.append(text)
                boxes_1000.append([x1, y1, x2, y2])
                boxes_px.append(px_box)
                confidences.append(conf)

        elif self.engine in ("paddle", "paddle-v4", "paddle-vl"):
            img_np = np.array(image.convert("RGB"))
            result = self.paddle.ocr(img_np, cls=True)
            if result and result[0]:
                for line in result[0]:
                    if len(line) < 2:
                        continue
                    box = line[0]
                    text = line[1][0]
                    conf = float(line[1][1])
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    px_box = [float(x1), float(y1), float(x2), float(y2)]
                    nx1 = max(0, min(1000, int(1000 * x1 / width)))
                    ny1 = max(0, min(1000, int(1000 * y1 / height)))
                    nx2 = max(0, min(1000, int(1000 * x2 / width)))
                    ny2 = max(0, min(1000, int(1000 * y2 / height)))
                    words.append(text)
                    boxes_1000.append([nx1, ny1, nx2, ny2])
                    boxes_px.append(px_box)
                    confidences.append(conf)

        elif self.engine == "doctr":
            img_np = np.array(image.convert("RGB"))
            # docTR accepts a list of numpy pages
            result = self.doctr([img_np])
            export = result.export()
            for page in export.get("pages", []):
                for block in page.get("blocks", []):
                    for line in block.get("lines", []):
                        for word in line.get("words", []):
                            text = (word.get("value") or "").strip()
                            if not text:
                                continue
                            conf = float(word.get("confidence", 0.0))
                            geom = word.get("geometry") or []
                            if len(geom) != 2:
                                continue
                            (xmin, ymin), (xmax, ymax) = geom
                            x1, y1 = xmin * width, ymin * height
                            x2, y2 = xmax * width, ymax * height
                            px_box = [float(x1), float(y1), float(x2), float(y2)]
                            nx1 = max(0, min(1000, int(1000 * x1 / width)))
                            ny1 = max(0, min(1000, int(1000 * y1 / height)))
                            nx2 = max(0, min(1000, int(1000 * x2 / width)))
                            ny2 = max(0, min(1000, int(1000 * y2 / height)))
                            words.append(text)
                            boxes_1000.append([nx1, ny1, nx2, ny2])
                            boxes_px.append(px_box)
                            confidences.append(conf)

        return words, boxes_1000, confidences, boxes_px


# ===========================================================================
# Utility helpers
# ===========================================================================

def _calculate_iou(boxA, boxB):
    """IoU between two [x1, y1, x2, y2] boxes (0-1000 scale)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / float(areaA + areaB - inter + 1e-6)


def _normalize_box(box, width, height):
    """Convert pixel-coords [x1,y1,x2,y2] → 0-1000 normalised."""
    return [
        max(0, min(1000, int(1000 * box[0] / width))),
        max(0, min(1000, int(1000 * box[1] / height))),
        max(0, min(1000, int(1000 * box[2] / width))),
        max(0, min(1000, int(1000 * box[3] / height))),
    ]


def _denoise_image(image: Image.Image):
    """
    Light denoise to help OCR:
      • If OpenCV is available: fastNlMeansDenoisingColored.
      • Else: 3x3 median filter via PIL.
    """
    if _HAS_CV2:
        img_np = np.array(image.convert("RGB"))
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
        )
        return Image.fromarray(denoised)
    # PIL fallback
    from PIL import ImageFilter
    return image.convert("RGB").filter(ImageFilter.MedianFilter(size=3))


def _sort_key(box):
    """Top-to-bottom, left-to-right reading order key."""
    cy = (box[1] + box[3]) / 2.0
    cx = (box[0] + box[2]) / 2.0
    # Group into lines: quantise y by ~3% of the 1000-scale
    line = int(cy / 30)
    return (line, cx)


def _sort_tokens(words, boxes):
    """Sort word-box pairs by reading order. Returns sorted copies."""
    pairs = list(zip(words, boxes))
    pairs.sort(key=lambda p: _sort_key(p[1]))
    if not pairs:
        return [], []
    sorted_words, sorted_boxes = zip(*pairs)
    return list(sorted_words), list(sorted_boxes)


def _normalize_text(text):
    """Normalise text for fair OCR comparison:
    lowercase, strip punctuation/special symbols, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation & symbols
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _fuzzy_ratio(a, b):
    """Sequence-matcher similarity ratio (0-1) on normalised text."""
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


# ===========================================================================
# FUNSD Dataset Loader
# ===========================================================================

def load_funsd_documents(data_dir, split="testing_data"):
    """
    Load FUNSD documents from the local filesystem.
    
    Returns list of dicts:
        {
            "doc_id": str,
            "image_path": str,
            "gt_words": List[str],
            "gt_boxes": List[List[int]],   # pixel-scale
        }
    """
    ann_dir = os.path.join(data_dir, split, "annotations")
    img_dir = os.path.join(data_dir, split, "images")

    documents = []
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
        doc_id = os.path.splitext(os.path.basename(ann_path))[0]
        img_path = os.path.join(img_dir, doc_id + ".png")

        if not os.path.exists(img_path):
            print(f"  [WARN] Image not found for {doc_id}, skipping.")
            continue

        with open(ann_path, "r") as f:
            data = json.load(f)

        gt_words = []
        gt_boxes = []

        for entity in data.get("form", []):
            for w in entity.get("words", []):
                text = w.get("text", "").strip()
                box = w.get("box", [0, 0, 0, 0])
                if text:
                    gt_words.append(text)
                    gt_boxes.append(box)

        documents.append({
            "doc_id": doc_id,
            "image_path": img_path,
            "gt_words": gt_words,
            "gt_boxes": gt_boxes,
        })

    return documents


# ===========================================================================
# OCR Evaluator
# ===========================================================================

class OCREvaluator:
    """Quantitative OCR evaluation against FUNSD ground-truth."""

    def __init__(self, ocr_engine="tesseract", denoise=False):
        self.ocr_engine = ocr_engine
        self.denoise = denoise
        self.backend = FastOCRBackend(engine=ocr_engine)

    # -----------------------------------------------------------------
    # Per-document evaluation
    # -----------------------------------------------------------------
    def evaluate_document(self, image_path, gt_words, gt_boxes):
        """
        Evaluate a single document.

        Args:
            image_path: path to the document image.
            gt_words:   list of ground-truth word strings.
            gt_boxes:   list of GT bounding boxes in pixel scale.

        Returns:
            dict with all metric groups.
        """
        image = Image.open(image_path).convert("RGB")
        if self.denoise:
            image = _denoise_image(image)
        width, height = image.size

        # --- Run OCR ---
        ocr_words, ocr_boxes_1000, _, _ = self.backend.run(image)

        # --- Normalise GT boxes to 0-1000 ---
        gt_boxes_1000 = [_normalize_box(b, width, height) for b in gt_boxes]

        # --- Sort both token lists by reading order ---
        gt_words_s, gt_boxes_s = _sort_tokens(gt_words, gt_boxes_1000)
        ocr_words_s, ocr_boxes_s = _sort_tokens(ocr_words, ocr_boxes_1000)

        # --- 1. Character-level metrics (CER / WER) ---
        char_metrics = self._compute_char_metrics(gt_words_s, ocr_words_s)

        # --- 2. Word recall ---
        recall_metrics = self._compute_word_recall(
            gt_words_s, gt_boxes_s, ocr_words_s, ocr_boxes_s
        )

        # --- 3. Segmentation errors ---
        seg_metrics = self._compute_segmentation(
            gt_words_s, gt_boxes_s, ocr_words_s, ocr_boxes_s
        )

        # --- 4. Spatial accuracy ---
        spatial_metrics = self._compute_spatial_accuracy(
            gt_words_s, gt_boxes_s, ocr_words_s, ocr_boxes_s
        )

        counts_metrics = {
            "gt_words": len(gt_words_s),
            "ocr_words": len(ocr_words_s),
            "ocr_to_gt_ratio": round(len(ocr_words_s) / max(len(gt_words_s), 1), 4),
        }

        return {
            "character": char_metrics,
            "recall": recall_metrics,
            "segmentation": seg_metrics,
            "spatial": spatial_metrics,
            "counts": counts_metrics,
        }

    # -----------------------------------------------------------------
    # Dataset-level evaluation
    # -----------------------------------------------------------------
    def evaluate_dataset(self, documents):
        """
        Evaluate all documents and aggregate results.
        
        Args:
            documents: list of dicts from load_funsd_documents().

        Returns:
            dict with per-document and overall results.
        """
        per_doc = {}
        # Accumulators for weighted averaging
        total_chars = 0
        total_words = 0
        total_char_errors = 0
        total_word_errors = 0
        total_gt_words = 0
        total_ocr_words = 0
        total_matched = 0
        total_missed = 0
        over_seg_counts = []
        under_seg_counts = []
        order_scores = []
        all_ious = []
        ocr_words_per_doc = []
        gt_words_per_doc = []

        for i, doc in enumerate(documents):
            doc_id = doc["doc_id"]
            print(f"  [{i+1}/{len(documents)}] Evaluating {doc_id} ...", end=" ")

            result = self.evaluate_document(
                doc["image_path"], doc["gt_words"], doc["gt_boxes"]
            )
            per_doc[doc_id] = result
            print(f"CER={result['character']['cer']:.3f}  "
                  f"Recall={result['recall']['recall']:.2f}")

            # Accumulate
            c = result["character"]
            total_chars += c["total_characters"]
            total_words += c["total_words"]
            total_char_errors += c["cer"] * c["total_characters"]
            total_word_errors += c["wer"] * c["total_words"]

            r = result["recall"]
            total_gt_words += r["gt_words"]
            total_matched += r["matched_words"]
            total_missed += r["missed_words"]

            counts = result.get("counts", {})
            ocr_count = counts.get("ocr_words", 0)
            total_ocr_words += ocr_count
            ocr_words_per_doc.append(ocr_count)
            gt_words_per_doc.append(counts.get("gt_words", 0))

            s = result["segmentation"]
            over_seg_counts.append(s["over_segmentation_rate"])
            under_seg_counts.append(s["under_segmentation_rate"])
            order_scores.append(s["order_consistency_score"])

            all_ious.extend(result["spatial"].get("matched_ious", []))

        # --- Overall aggregation ---
        overall_cer = total_char_errors / max(total_chars, 1)
        overall_wer = total_word_errors / max(total_words, 1)
        overall_recall = total_matched / max(total_gt_words, 1)
        mean_over_seg = statistics.mean(over_seg_counts) if over_seg_counts else 0
        mean_under_seg = statistics.mean(under_seg_counts) if under_seg_counts else 0
        mean_order = statistics.mean(order_scores) if order_scores else 1.0

        if all_ious:
            mean_iou = statistics.mean(all_ious)
            median_iou = statistics.median(all_ious)
            pct_iou_above_07 = sum(1 for v in all_ious if v > 0.7) / len(all_ious)
        else:
            mean_iou = 0.0
            median_iou = 0.0
            pct_iou_above_07 = 0.0

        overall = {
            "cer": round(overall_cer, 4),
            "wer": round(overall_wer, 4),
            "word_recall": round(overall_recall, 4),
            "over_segmentation": round(mean_over_seg, 4),
            "under_segmentation": round(mean_under_seg, 4),
            "order_consistency": round(mean_order, 4),
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "pct_iou_above_0.7": round(pct_iou_above_07, 4),
            "total_documents": len(documents),
            "total_gt_words": total_gt_words,
            "total_ocr_words": total_ocr_words,
            "ocr_to_gt_ratio": round(total_ocr_words / max(total_gt_words, 1), 4),
            "mean_ocr_words_per_doc": round(statistics.mean(ocr_words_per_doc), 2) if ocr_words_per_doc else 0,
            "median_ocr_words_per_doc": round(statistics.median(ocr_words_per_doc), 2) if ocr_words_per_doc else 0,
            "mean_gt_words_per_doc": round(statistics.mean(gt_words_per_doc), 2) if gt_words_per_doc else 0,
            "total_matched_words": total_matched,
            "total_missed_words": total_missed,
        }

        return {"overall": overall, "per_document": per_doc}

    # =================================================================
    # Metric implementations
    # =================================================================

    def _compute_char_metrics(self, gt_words, ocr_words):
        """CER and WER via jiwer over full-document text."""
        gt_text = " ".join(gt_words) if gt_words else ""
        ocr_text = " ".join(ocr_words) if ocr_words else ""

        if not gt_text:
            return {"cer": 0.0, "wer": 0.0, "total_characters": 0, "total_words": 0}

        # jiwer requires non-empty hypothesis; use a placeholder if OCR is empty
        if not ocr_text:
            ocr_text = " "

        cer_val = compute_cer(gt_text, ocr_text)
        wer_val = compute_wer(gt_text, ocr_text)

        return {
            "cer": round(min(cer_val, 1.0), 4),
            "wer": round(min(wer_val, 2.0), 4),  # WER can exceed 1.0
            "total_characters": len(gt_text.replace(" ", "")),
            "total_words": len(gt_words),
        }

    def _compute_word_recall(self, gt_words, gt_boxes, ocr_words, ocr_boxes):
        """
        For each GT word, check if OCR detected a matching word
        (fuzzy ≥ 0.6 AND spatial IoU > 0.3, or exact text match nearby).
        Text is normalised (lowercase, no punctuation) before comparison.
        """
        matched = 0
        missed_list = []
        ocr_used = [False] * len(ocr_words)

        for gi, (gw, gb) in enumerate(zip(gt_words, gt_boxes)):
            best_score = 0.0
            best_idx = -1

            for oi, (ow, ob) in enumerate(zip(ocr_words, ocr_boxes)):
                if ocr_used[oi]:
                    continue

                iou = _calculate_iou(gb, ob)
                text_sim = _fuzzy_ratio(gw, ow)

                # Match criteria: decent text similarity + some spatial overlap
                # OR exact normalised match with moderate proximity
                if text_sim >= 0.6 and iou > 0.3:
                    score = text_sim + iou
                elif text_sim >= 0.95 and iou > 0.1:
                    score = 2.0 + iou
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_idx = oi

            if best_idx >= 0:
                matched += 1
                ocr_used[best_idx] = True
            else:
                missed_list.append({"word": gw, "box": gb})

        gt_count = len(gt_words)
        return {
            "gt_words": gt_count,
            "matched_words": matched,
            "missed_words": gt_count - matched,
            "recall": round(matched / max(gt_count, 1), 4),
            "missed_word_list": missed_list,
        }

    def _compute_segmentation(self, gt_words, gt_boxes, ocr_words, ocr_boxes):
        """
        Detect over-segmentation, under-segmentation, and order mismatch.
        Uses IoU > 0.5 for box matching.
        """
        iou_threshold = 0.5

        # Build GT→OCR and OCR→GT match maps
        gt_to_ocr = {i: [] for i in range(len(gt_words))}
        ocr_to_gt = {i: [] for i in range(len(ocr_words))}

        for gi, gb in enumerate(gt_boxes):
            for oi, ob in enumerate(ocr_boxes):
                iou = _calculate_iou(gb, ob)
                if iou > iou_threshold:
                    gt_to_ocr[gi].append(oi)
                    ocr_to_gt[oi].append(gi)

        # ---- A) Over-segmentation ----
        # GT word matched by ≥2 OCR tokens (word was split)
        over_seg_count = sum(1 for matches in gt_to_ocr.values() if len(matches) >= 2)
        over_seg_rate = over_seg_count / max(len(gt_words), 1)

        # ---- B) Under-segmentation ----
        # OCR token matched by ≥2 GT words (words were merged)
        under_seg_count = sum(1 for matches in ocr_to_gt.values() if len(matches) >= 2)
        under_seg_rate = under_seg_count / max(len(ocr_words), 1)

        # ---- C) Order mismatch ----
        # For GT words that have exactly 1 OCR match, check if OCR indices
        # preserve the GT ordering (Kendall-tau-like metric).
        gt_order_pairs = []
        for gi in sorted(gt_to_ocr.keys()):
            matches = gt_to_ocr[gi]
            if len(matches) == 1:
                gt_order_pairs.append((gi, matches[0]))

        if len(gt_order_pairs) >= 2:
            concordant = 0
            discordant = 0
            for i in range(len(gt_order_pairs)):
                for j in range(i + 1, len(gt_order_pairs)):
                    gi1, oi1 = gt_order_pairs[i]
                    gi2, oi2 = gt_order_pairs[j]
                    # GT order: gi1 < gi2 always (sorted)
                    if oi1 < oi2:
                        concordant += 1
                    elif oi1 > oi2:
                        discordant += 1
            total_pairs = concordant + discordant
            order_score = concordant / max(total_pairs, 1)
        else:
            order_score = 1.0  # Not enough data → assume correct

        return {
            "over_segmentation_count": over_seg_count,
            "over_segmentation_rate": round(over_seg_rate, 4),
            "under_segmentation_count": under_seg_count,
            "under_segmentation_rate": round(under_seg_rate, 4),
            "order_consistency_score": round(order_score, 4),
        }

    def _compute_spatial_accuracy(self, gt_words, gt_boxes, ocr_words, ocr_boxes):
        """
        For matched word pairs, compute IoU statistics.
        Matching: best IoU per GT word (greedy, > 0.1 threshold).
        """
        matched_ious = []
        ocr_used = [False] * len(ocr_boxes)

        for gi, (gw, gb) in enumerate(zip(gt_words, gt_boxes)):
            best_iou = 0.0
            best_idx = -1

            for oi, (ow, ob) in enumerate(zip(ocr_words, ocr_boxes)):
                if ocr_used[oi]:
                    continue
                iou = _calculate_iou(gb, ob)
                text_sim = _fuzzy_ratio(gw, ow)
                # Require some text similarity to prevent matching wrong boxes
                if iou > best_iou and text_sim >= 0.4:
                    best_iou = iou
                    best_idx = oi

            if best_idx >= 0 and best_iou > 0.1:
                matched_ious.append(round(best_iou, 4))
                ocr_used[best_idx] = True

        if matched_ious:
            mean_iou = statistics.mean(matched_ious)
            median_iou = statistics.median(matched_ious)
            pct_above_07 = sum(1 for v in matched_ious if v > 0.7) / len(matched_ious)
        else:
            mean_iou = 0.0
            median_iou = 0.0
            pct_above_07 = 0.0

        return {
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "pct_iou_above_0.7": round(pct_above_07, 4),
            "num_matched": len(matched_ious),
            "matched_ious": matched_ious,
        }


# ===========================================================================
# Report printing / saving
# ===========================================================================

def print_summary(engine, overall):
    """Pretty-print the evaluation summary."""
    print()
    print(f"OCR ENGINE: {engine}")
    print("-" * 42)
    print(f"  CER:                 {overall['cer']:.4f}")
    print(f"  WER:                 {overall['wer']:.4f}")
    print(f"  Word Recall:         {overall['word_recall']:.4f}")
    print(f"  Over-segmentation:   {overall['over_segmentation']:.4f}")
    print(f"  Under-segmentation:  {overall['under_segmentation']:.4f}")
    print(f"  Order Consistency:   {overall['order_consistency']:.4f}")
    print(f"  Mean IoU:            {overall['mean_iou']:.4f}")
    print(f"  Median IoU:          {overall['median_iou']:.4f}")
    print(f"  % IoU > 0.7:         {overall['pct_iou_above_0.7']:.4f}")
    print("-" * 42)
    print(f"  Documents evaluated: {overall['total_documents']}")
    print(f"  GT words:            {overall['total_gt_words']}")
    print(f"  OCR words:           {overall['total_ocr_words']}")
    print(f"  OCR/GT ratio:        {overall['ocr_to_gt_ratio']:.4f}")
    print(f"  Avg OCR words/doc:   {overall['mean_ocr_words_per_doc']:.2f}")
    print(f"  Median OCR words/doc:{overall['median_ocr_words_per_doc']:.2f}")
    print(f"  Matched words:       {overall['total_matched_words']}")
    print(f"  Missed words:        {overall['total_missed_words']}")
    print("-" * 42)
    print()


def save_report(engine, report, output_dir="."):
    """Save full JSON report (strip matched_ious lists for size)."""
    # Deep-copy and strip large lists for the per-doc section
    clean = {"overall": report["overall"], "per_document": {}}
    for doc_id, doc_data in report["per_document"].items():
        clean_doc = {}
        for section, metrics in doc_data.items():
            clean_metrics = {k: v for k, v in metrics.items()
                            if k not in ("missed_word_list", "matched_ious")}
            clean_doc[section] = clean_metrics
        clean["per_document"][doc_id] = clean_doc

    path = os.path.join(output_dir, f"ocr_eval_report_{engine}.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Report saved → {path}")


def print_comparison(results):
    """Print side-by-side comparison table for multiple engines."""
    engines = list(results.keys())
    metrics = [
        ("CER", "cer"),
        ("WER", "wer"),
        ("Word Recall", "word_recall"),
        ("Over-segmentation", "over_segmentation"),
        ("Under-segmentation", "under_segmentation"),
        ("Order Consistency", "order_consistency"),
        ("Mean IoU", "mean_iou"),
        ("Median IoU", "median_iou"),
        ("% IoU > 0.7", "pct_iou_above_0.7"),
    ]

    # Header
    header = f"{'Metric':<22}"
    for eng in engines:
        header += f"  {eng:>12}"
    print()
    print("=" * (22 + 14 * len(engines)))
    print(header)
    print("-" * (22 + 14 * len(engines)))

    for label, key in metrics:
        row = f"{label:<22}"
        for eng in engines:
            val = results[eng]["overall"].get(key, 0)
            row += f"  {val:>12.4f}"
        print(row)

    print("=" * (22 + 14 * len(engines)))
    print()


# ===========================================================================
# CLI entry point
# ===========================================================================

def _load_splits(data_dir, split, limit=None):
    """Load documents from one or both splits."""
    if split == "all":
        splits = ["training_data", "testing_data"]
    else:
        splits = [split]

    all_docs = []
    for s in splits:
        print(f"Loading FUNSD documents from {data_dir}/{s} ...")
        docs = load_funsd_documents(data_dir, split=s)
        print(f"  → {len(docs)} documents from {s}")
        all_docs.extend(docs)

    if limit is not None and limit > 0:
        all_docs = all_docs[:limit]
        print(f"Limiting to first {len(all_docs)} documents.\n")

    print(f"Total: {len(all_docs)} documents.\n")
    return all_docs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR quality against FUNSD ground-truth."
    )
    parser.add_argument(
        "--ocr_engine", type=str, default="tesseract",
        choices=["tesseract", "paddle", "doctr"],
        help="OCR engine to evaluate (default: tesseract)"
    )
    parser.add_argument(
        "--denoise", action="store_true",
        help="Apply light denoising to images before OCR"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to FUNSD root directory (contains testing_data/)"
    )
    parser.add_argument(
        "--split", type=str, default="testing_data",
        choices=["training_data", "testing_data", "all"],
        help="Dataset split to evaluate (default: testing_data)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both Tesseract and Paddle, print comparison"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of documents to evaluate (e.g., 199)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save JSON reports (default: cwd)"
    )
    args = parser.parse_args()

    start_time = time.time()

    # Load FUNSD documents (both splits by default)
    documents = _load_splits(args.data_dir, args.split, limit=args.limit)

    if not documents:
        print("ERROR: No documents found. Check --data_dir path.")
        return

    if args.compare:
        # --- Comparison mode ---
        all_results = {}
        for engine in ["tesseract", "paddle"]:
            print(f"\n{'='*50}")
            print(f"  Running evaluation: {engine.upper()}")
            print(f"{'='*50}")
            evaluator = OCREvaluator(ocr_engine=engine, denoise=args.denoise)
            report = evaluator.evaluate_dataset(documents)
            all_results[engine] = report

            print_summary(engine, report["overall"])
            save_report(engine, report, args.output_dir)

        print_comparison(all_results)

    else:
        # --- Single engine mode ---
        evaluator = OCREvaluator(ocr_engine=args.ocr_engine, denoise=args.denoise)
        report = evaluator.evaluate_dataset(documents)

        print_summary(args.ocr_engine, report["overall"])
        save_report(args.ocr_engine, report, args.output_dir)

    elapsed = time.time() - start_time
    print(f"⏱  Total time: {elapsed:.1f}s ({elapsed/max(len(documents),1):.2f}s/doc)")


if __name__ == "__main__":
    main()
