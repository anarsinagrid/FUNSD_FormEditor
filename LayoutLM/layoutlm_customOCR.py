"""
layoutlm_customOCR.py
=====================
LayoutLMv3 OCR-Consistent Training Pipeline and Inference abstraction.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
# Optional DocFormer imports (only needed when arch=docformer)
try:
    from transformers import DocFormerProcessor, DocFormerForTokenClassification
except ImportError:
    DocFormerProcessor = None
    DocFormerForTokenClassification = None
import torch.nn as nn
from collections import Counter
from torch.optim import AdamW

# ==============================================================================
# 1. OCR Abstraction Layer
# ==============================================================================
class OCRBackend:
    def __init__(self, engine="tesseract"):
        self.engine = engine
        if self.engine == "paddle":
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                print("Warning: PaddleOCR not installed.")
                self.paddle = None
            else:
                self.paddle = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
        elif self.engine == "tesseract":
            try:
                import pytesseract
                # Explicitly set path for homebrew Apple Silicon if necessary
                if os.path.exists("/opt/homebrew/bin/tesseract"):
                    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
                self.pytesseract = pytesseract
            except ImportError:
                print("Warning: pytesseract not installed.")
        elif self.engine == "doctr":
            try:
                from doctr.models import ocr_predictor
            except ImportError:
                print("Warning: python-doctr not installed.")
                self.doctr = None
            else:
                self.doctr_device = torch.device(
                    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self.doctr = ocr_predictor(
                    pretrained=True,
                    assume_straight_pages=True,
                    export_as_straight_boxes=True,
                )
                self.doctr.to(self.doctr_device)
                print(f"docTR device: {self.doctr_device}")

    def run(self, image: Image.Image):
        """
        Runs OCR on the provided PIL Image.
        Returns:
            words (List[str]): Extracted words.
            boxes_1000 (List[List[int]]): Bounding boxes normalized to 0-1000 scale.
            confidences (List[float]): Optional confidence scores.
            boxes_px (List[List[float]]): Bounding boxes in original pixel scale.
        """
        width, height = image.size
        words = []
        boxes_1000 = []
        boxes_px = []
        confidences = []

        if self.engine == "tesseract":
            data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                conf = float(data['conf'][i])
                
                # Original pixel scale
                px_box = [float(x), float(y), float(x + w), float(y + h)]
                
                # Normalize box to 0-1000 LayoutLM scale
                x1 = int(1000 * x / width)
                y1 = int(1000 * y / height)
                x2 = int(1000 * (x + w) / width)
                y2 = int(1000 * (y + h) / height)
                
                x1, y1 = max(0, min(1000, x1)), max(0, min(1000, y1))
                x2, y2 = max(0, min(1000, x2)), max(0, min(1000, y2))
                
                words.append(text)
                boxes_1000.append([x1, y1, x2, y2])
                boxes_px.append(px_box)
                confidences.append(conf)

        elif self.engine == "paddle":
            if not self.paddle:
                raise RuntimeError("PaddleOCR not properly initialized.")
            img_np = np.array(image.convert("RGB"))
            
            # Use paddle engine natively
            result = self.paddle.ocr(img_np, cls=True)
            
            if result and result[0]:
                res = result[0]
                for line in res:
                    if len(line) < 2:
                        continue
                    box = line[0]        # [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    text = line[1][0]
                    conf = float(line[1][1])
                    
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    px_box = [float(x1), float(y1), float(x2), float(y2)]
                    
                    # Normalize box to 0-1000 LayoutLM scale
                    nx1 = int(1000 * x1 / width)
                    ny1 = int(1000 * y1 / height)
                    nx2 = int(1000 * x2 / width)
                    ny2 = int(1000 * y2 / height)
                    
                    nx1, ny1 = max(0, min(1000, nx1)), max(0, min(1000, ny1))
                    nx2, ny2 = max(0, min(1000, nx2)), max(0, min(1000, ny2))
                    
                    words.append(text)
                    boxes_1000.append([nx1, ny1, nx2, ny2])
                    boxes_px.append(px_box)
                    confidences.append(conf)

        elif self.engine == "doctr":
            if not self.doctr:
                raise RuntimeError("docTR not properly initialized.")
            img_np = np.array(image.convert("RGB"))
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
                            nx1 = int(1000 * x1 / width)
                            ny1 = int(1000 * y1 / height)
                            nx2 = int(1000 * x2 / width)
                            ny2 = int(1000 * y2 / height)
                            nx1, ny1 = max(0, min(1000, nx1)), max(0, min(1000, ny1))
                            nx2, ny2 = max(0, min(1000, nx2)), max(0, min(1000, ny2))
                            words.append(text)
                            boxes_1000.append([nx1, ny1, nx2, ny2])
                            boxes_px.append(px_box)
                            confidences.append(conf)

        return words, boxes_1000, confidences, boxes_px

# ==============================================================================
# 2. Dataset Alignment logic
# ==============================================================================
def _calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def _get_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

# Minimum IoU an OCR box must overlap a GT box to claim its label.
# Anything below this gets assigned "O" instead of a weak spatial match.
_MIN_LABEL_IOU = 0.5

# Radius (in LayoutLM 0‑1000 space) within which Levenshtein fallback is allowed.
_LEV_PROXIMITY_RADIUS = 60

def align_ocr_to_ground_truth(ocr_words, ocr_boxes_1000, gt_words, gt_boxes_1000, gt_labels, default_label_id=0):
    """
    Matches each OCR word to the best *unmatched* ground-truth word.

    Key properties:
      * Each GT word can be claimed by at most one OCR word (deduplication via
        `used_gt_indices`). This prevents duplicate B-tags and entity bleeding.
      * IoU must be >= _MIN_LABEL_IOU for a spatial match to be accepted.
        Sub-threshold overlaps do NOT inherit the GT label.
      * Levenshtein fallback fires only when the best IoU across ALL GT words
        is below the threshold, keeping it as a last resort for highly messy OCR.
    """
    import Levenshtein

    aligned_labels = []

    # Pre-calculate centers once for speed
    gt_centers = [_get_center(box) for box in gt_boxes_1000]

    # Track which GT words have already been consumed to prevent double-matching.
    used_gt_indices = set()

    for ocr_idx, ocr_box in enumerate(ocr_boxes_1000):
        ocr_word = ocr_words[ocr_idx].lower().strip()
        ocr_center = _get_center(ocr_box)

        best_iou = 0.0
        best_gt_idx = -1

        # 1. Primary: highest IoU among *unused* GT words
        for i, gt_box in enumerate(gt_boxes_1000):
            if i in used_gt_indices:
                continue
            iou = _calculate_iou(ocr_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        # 2. Accept only if the best IoU meets the minimum threshold.
        #    If not, try Levenshtein fallback among nearby *unused* GT words.
        if best_iou < _MIN_LABEL_IOU:
            best_gt_idx = -1   # reset — spatial match was too weak
            best_dist = float('inf')

            for i, gt_box in enumerate(gt_boxes_1000):
                if i in used_gt_indices:
                    continue
                cx, cy = gt_centers[i]
                spatial_dist = ((ocr_center[0] - cx) ** 2 + (ocr_center[1] - cy) ** 2) ** 0.5

                if spatial_dist < _LEV_PROXIMITY_RADIUS:   # only consider nearby GT words
                    gt_word = gt_words[i].lower().strip()
                    dist = Levenshtein.distance(ocr_word, gt_word)

                    if dist < best_dist and dist <= 2:   # max 2 edit ops
                        best_dist = dist
                        best_gt_idx = i

                    # Containment fallback when text match also fails
                    elif (best_gt_idx == -1 and
                          gt_box[0] <= ocr_center[0] <= gt_box[2] and
                          gt_box[1] <= ocr_center[1] <= gt_box[3]):
                        best_gt_idx = i

        if best_gt_idx != -1:
            used_gt_indices.add(best_gt_idx)   # mark consumed
            aligned_labels.append(gt_labels[best_gt_idx])
        else:
            aligned_labels.append(default_label_id)

    return aligned_labels

# ==============================================================================
# 3. Training Logic
# ==============================================================================
def compute_metrics(p, label_list, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def get_layerwise_lr_decay_optimizer(model, base_lr=1.5e-5, layer_decay=0.9, weight_decay=0.01):
    lm = model.layoutlmv3
    num_layers = len(lm.encoder.layer)
    optimizer_grouped_parameters = []
    
    # Layer 0: Embeddings
    lr = base_lr * (layer_decay ** (num_layers + 1))
    optimizer_grouped_parameters.append({"params": lm.embeddings.parameters(), "lr": lr, "weight_decay": weight_decay})
    if hasattr(lm, "patch_embed"):
        optimizer_grouped_parameters.append({"params": lm.patch_embed.parameters(), "lr": lr, "weight_decay": weight_decay})

    # Intermediate Layers
    for layer_idx in range(num_layers):
        layer = lm.encoder.layer[layer_idx]
        lr = base_lr * (layer_decay ** (num_layers - layer_idx))
        optimizer_grouped_parameters.append({"params": layer.parameters(), "lr": lr, "weight_decay": weight_decay})

    # Top Layer: Classifier head (full LR)
    optimizer_grouped_parameters.append({"params": model.classifier.parameters(), "lr": base_lr, "weight_decay": weight_decay})
    return AdamW(optimizer_grouped_parameters)

def get_optimizer(model, arch="layoutlmv3", base_lr=1.5e-5, layer_decay=0.9, weight_decay=0.01):
    """
    Architecture-aware optimizer builder.
    """
    if arch == "layoutlmv3" and hasattr(model, "layoutlmv3"):
        return get_layerwise_lr_decay_optimizer(model, base_lr=base_lr, layer_decay=layer_decay, weight_decay=weight_decay)
    # Fallback: flat AdamW
    return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

def freeze_backbone_layers(model, num_frozen_layers=6):
    if not hasattr(model, "layoutlmv3"):
        return
    lm = model.layoutlmv3
    # Freeze embeddings and patch embedding
    if hasattr(lm, "embeddings"):
        for p in lm.embeddings.parameters():
            p.requires_grad = False
    if hasattr(lm, "patch_embed"):
        for p in lm.patch_embed.parameters():
            p.requires_grad = False
    # Freeze first N transformer blocks
    if hasattr(lm, "encoder") and hasattr(lm.encoder, "layer"):
        for layer in lm.encoder.layer[:num_frozen_layers]:
            for p in layer.parameters():
                p.requires_grad = False

def interpolate_pos_encoding(model, height, width):
    """
    Interpolates visual position embeddings to support higher resolutions (e.g., 384x384).
    """
    import torch.nn.functional as F
    pos_embed = model.layoutlmv3.pos_embed # Shape: (1, 197, 768)
    n, p, d = pos_embed.shape
    
    cls_pos_embed = pos_embed[:, 0:1, :]
    patch_pos_embed = pos_embed[:, 1:, :]
    
    dim = d
    old_h = old_w = int((p - 1) ** 0.5)
    new_h = height // 16
    new_w = width // 16
    
    if old_h == new_h and old_w == new_w:
        return
        
    print(f"Interpolating visual position embeddings from {old_h}x{old_w} to {new_h}x{new_w}")
    patch_pos_embed = patch_pos_embed.reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2)
    patch_pos_embed = F.interpolate(patch_pos_embed, size=(new_h, new_w), mode='bicubic', align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
    
    new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
    model.layoutlmv3.pos_embed = nn.Parameter(new_pos_embed)

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build loss function ONCE here — constructing nn.CrossEntropyLoss every
        # forward step was allocating a new CUDA/MPS kernel graph each time.
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
            self._loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self._loss_fct = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self._loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class MPSCachePurgeCallback(TrainerCallback):
    """
    Clears the MPS memory cache after every evaluation pass.
    Without this, PyTorch/MPS holds onto eval-allocated buffers and the
    training loop progressively slows down after epoch boundaries.
    """
    def on_evaluate(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

class StepTimerCallback(TrainerCallback):
    def __init__(self, log_every=50, sync_mps=False, log_mps_mem=False):
        self.log_every = log_every
        self.sync_mps = sync_mps
        self.log_mps_mem = log_mps_mem
        self._t0 = None
        self._ema = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is None:
            return
        if self.sync_mps and torch.backends.mps.is_available():
            torch.mps.synchronize()
        dt = time.perf_counter() - self._t0
        if self._ema is None:
            self._ema = dt
        else:
            self._ema = 0.9 * self._ema + 0.1 * dt
        if self.log_every and state.global_step % self.log_every == 0:
            extra = ""
            if self.log_mps_mem and torch.backends.mps.is_available():
                try:
                    alloc = torch.mps.current_allocated_memory() / (1024 ** 2)
                    driver = torch.mps.driver_allocated_memory() / (1024 ** 2)
                    extra = f" mps_alloc={alloc:.1f}MB mps_driver={driver:.1f}MB"
                except Exception:
                    pass
            print(f"[speed] step_time={dt:.3f}s ema={self._ema:.3f}s step={state.global_step}{extra}")

class TrainingMonitorCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_f1" in metrics:
            best = state.best_metric
            best_str = f"{best:.4f}" if best is not None else "n/a"
            print(f"[monitor] eval_f1={metrics['eval_f1']:.4f} best_f1={best_str} epoch={state.epoch:.2f}")

    def on_train_end(self, args, state, control, **kwargs):
        if state.best_model_checkpoint:
            print(f"[monitor] best_checkpoint={state.best_model_checkpoint}")
        if state.best_metric is not None:
            print(f"[monitor] best_metric={state.best_metric:.4f}")

def train(
    ocr_engine="tesseract",
    arch="layoutlmv3",
    model_size="base",
    speed_log_steps=50,
    speed_sync_mps=False,
    speed_log_mps_mem=False,
    early_stopping_patience=5,
    cache_dir=None,
    gradient_checkpointing=True,
    init_checkpoint=None,
    min_label_iou=_MIN_LABEL_IOU,
    target_size=512,
    output_suffix=None,
):
    if not cache_dir:
        raise ValueError("cache_dir is required. Pass --cache_dir /absolute/path/to/hf_cache")

    # Allow CLI override of alignment strictness
    global _MIN_LABEL_IOU
    _MIN_LABEL_IOU = min_label_iou

    print(f"Starting Training Pipeline using OCR Engine: {ocr_engine}")
    print(f"Script path: {__file__}")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"LayoutLM device: {device}")
    
    from datasets import load_dataset
    import evaluate
    
    # 3.1 Load Base assets
    dataset = load_dataset("nielsr/funsd", cache_dir=cache_dir)
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    # Processor must EXPLICITLY use apply_ocr=False (layoutlmv3). For docformer use its processor.
    if arch == "docformer":
        if DocFormerProcessor is None or DocFormerForTokenClassification is None:
            raise ImportError("DocFormer not available in this transformers version. Install transformers>=4.37 and re-run.")
        model_id = "microsoft/docformer-base"
        base_checkpoint = init_checkpoint if init_checkpoint else model_id
        processor = DocFormerProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    else:
        model_id = "microsoft/layoutlmv3-large" if model_size == "large" else "microsoft/layoutlmv3-base"
        base_checkpoint = init_checkpoint if init_checkpoint else model_id
        processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=False, cache_dir=cache_dir)
    TARGET_SIZE = int(target_size)
    if TARGET_SIZE <= 0 or TARGET_SIZE % 16 != 0:
        raise ValueError(f"target_size must be a positive multiple of 16, got {TARGET_SIZE}")

    # Use aspect-ratio-preserving resize: the long side is scaled to TARGET_SIZE,
    # the short side is padded with white to keep the spatial layout intact.
    # Square stretching distorts portrait FUNSD images and hurts geometry.
    processor.image_processor.size = {"height": TARGET_SIZE, "width": TARGET_SIZE}
    processor.image_processor.do_resize = True
    processor.image_processor.do_pad = True
    backend = OCRBackend(engine=ocr_engine)
    
    # 3.2 Dataset Encoder Map function
    # Counter for silent truncation monitoring — logged at encode time.
    _truncation_hits = [0]

    def _pad_to_square(image, target: int = TARGET_SIZE) -> Image.Image:
        """
        Aspect-ratio-preserving resize + white padding to a square canvas.
        The long side is scaled to `target`; the short side is padded with white.
        OCR boxes are normalised to 0-1000 using the *original* image dimensions
        BEFORE this resize, so no box rescaling is needed here — the 0-1000
        coordinate system is independent of pixel resolution by definition.
        """
        w, h = image.size
        scale = target / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        # Center the resized image on the canvas
        offset_x = (target - new_w) // 2
        offset_y = (target - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def encode_custom_ocr(example):
        image = example["image"].convert("RGB")
        # Run OCR on the original image (boxes are normalised against original dims)
        words, boxes_1000, _, _ = backend.run(image)

        # Skip entirely empty OCR outputs — padding with dummy tokens adds noise
        if not words:
            words = [""]
            boxes_1000 = [[0, 0, 0, 0]]

        # Align ground truth labels geometrically to the newly OCR'd boxes.
        # Uses dedup + IoU >= _MIN_LABEL_IOU threshold.
        gt_boxes = example["bboxes"]
        gt_words = example["words"]
        gt_labels = example["ner_tags"]
        aligned_labels = align_ocr_to_ground_truth(
            words, boxes_1000, gt_words, gt_boxes, gt_labels, label2id.get("O", 0)
        )

        # Feed the aspect-ratio-preserved image to the processor so the visual
        # patches align with the (undistorted) spatial layout of the text.
        image_for_model = _pad_to_square(image, target=TARGET_SIZE)

        # Pass pure extracted + aligned text to HF processor.
        # NOTE: do_resize/do_pad are set on the processor above; we pass a
        # pre-processed square image so it goes through unchanged.
        encoding = processor(
            image_for_model,
            words,
            boxes=boxes_1000,
            word_labels=aligned_labels,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Truncation monitor: warn if any doc fills the full context window.
        if encoding["input_ids"].shape[-1] == 512 and encoding["input_ids"][0, -1] != 1:
            _truncation_hits[0] += 1

        return {k: v.squeeze() for k, v in encoding.items()}

    print("Encoding Dataset with Custom OCR... (This may take a while)")
    encoded_train = dataset["train"].map(encode_custom_ocr, remove_columns=dataset["train"].column_names)
    encoded_test  = dataset["test"].map(encode_custom_ocr, remove_columns=dataset["test"].column_names)

    if _truncation_hits[0] > 0:
        print(f"[truncation] {_truncation_hits[0]} docs hit max_length=512 and were truncated.")
    else:
        print("[truncation] No docs exceeded max_length=512 — no silent truncation.")

    print("Computing class weights from training distribution...")
    all_labels = []
    for example in encoded_train:
        all_labels.extend([l for l in example["labels"] if l != -100])
    
    label_counts = Counter(all_labels)
    total_valid_labels = sum(label_counts.values())
    num_classes = len(id2label)
    
    class_weights = [1.0] * num_classes
    for class_id in range(num_classes):
        if label_counts[class_id] > 0:
            class_weights[class_id] = total_valid_labels / (num_classes * label_counts[class_id])
    print(f"Computed Class Weights: {class_weights}")

    # 3.3 Model Initialization
    if arch == "docformer":
        if DocFormerForTokenClassification is None:
            raise ImportError("DocFormerForTokenClassification unavailable. Upgrade transformers (pip install 'transformers>=4.37').")
        model = DocFormerForTokenClassification.from_pretrained(
            base_checkpoint,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir,
        ).to(device)
    else:
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            base_checkpoint,
            ignore_mismatched_sizes=True,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir,
        ).to(device)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing.")
        except Exception as e:
            print(f"Gradient checkpointing not enabled: {e}")
    
    # Interpolate for target resolution
    if arch == "layoutlmv3":
        interpolate_pos_encoding(model, TARGET_SIZE, TARGET_SIZE)
        model.config.input_size = TARGET_SIZE
        if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "init_visual_bbox"):
            grid = TARGET_SIZE // 16
            model.layoutlmv3.init_visual_bbox(image_size=(grid, grid))

    # Save the custom property so inference knows
    model.config.ocr_engine = ocr_engine

    optimizer = get_optimizer(model, arch=arch, base_lr=1.5e-5, layer_decay=0.9)
    metric = evaluate.load("seqeval")
    
    run_tag = "adapted" if init_checkpoint else model_size
    if output_suffix:
        run_tag = f"{run_tag}-{output_suffix}"
    model_prefix = "docformer" if arch == "docformer" else "layoutlmv3"
    output_directory = f"./{model_prefix}-funsd-{ocr_engine}-{run_tag}"

    training_args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=20,              # focused adaptation window
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,           # keep best + most recent for crash safety
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=1.5e-5,
        # LR schedule — cosine + warmup keeps LR healthy through all epochs
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        # MPS / Apple-Silicon perf fixes
        dataloader_pin_memory=False,   # pin_memory is CUDA-only; causes overhead on MPS
        dataloader_num_workers=4,      # load batches in background threads, not main thread
        bf16=torch.backends.mps.is_available(),  # bf16 is natively supported on Apple Silicon M-series
        gradient_accumulation_steps=4,
    )

    trainer_kwargs = dict(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_test,
        compute_metrics=lambda p: compute_metrics(p, label_list, metric),
        optimizers=(optimizer, None),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            TrainingMonitorCallback(),
            MPSCachePurgeCallback(),   # clears MPS cache after every eval to prevent balloon slowdown
            StepTimerCallback(
                log_every=speed_log_steps,
                sync_mps=speed_sync_mps,
                log_mps_mem=speed_log_mps_mem,
            ),
        ],
    )

    # transformers API compatibility:
    # older versions accept `tokenizer`, newer versions expect `processing_class`.
    try:
        trainer = CustomTrainer(tokenizer=processor, **trainer_kwargs)
    except TypeError as e:
        if "unexpected keyword argument 'tokenizer'" not in str(e):
            raise
        trainer = CustomTrainer(processing_class=processor, **trainer_kwargs)

    trainer.train()

# ==============================================================================
# 4. Editor-compliant Inference Logic
# ==============================================================================
def _union_box_px(boxes):
    if not boxes: return [0.0, 0.0, 0.0, 0.0]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]

def run_inference(image_path, checkpoint_path, selected_ocr, model_arch=None):
    """
    Called by the editor. Given an image path, runs OCR matching the model's training,
    runs the model, and groups the standard BIO sequence back into JSON schema blocks.
    schema: {"blocks": [{"id": X, "text": "...", "bbox": [...], "label": "question"}], "links": [], "tables": []}
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    image = Image.open(image_path).convert("RGB")
    
    # 1. OCR on the original image (boxes normalised to 0-1000 against original dims)
    backend = OCRBackend(engine=selected_ocr)
    words, boxes_1000, confidences, boxes_px = backend.run(image)

    if not words:
        return {"blocks": [], "links": [], "tables": []}

    # Detect architecture if not provided
    if model_arch is None:
        model_arch = "docformer" if "docformer" in (checkpoint_path or "") else "layoutlmv3"

    # 2. Setup Inference Context
    if model_arch == "docformer":
        if DocFormerProcessor is None or DocFormerForTokenClassification is None:
            raise ImportError("DocFormer not available. Install transformers>=4.37 to use docformer inference.")
        processor = DocFormerProcessor.from_pretrained("microsoft/docformer-base")
        model = DocFormerForTokenClassification.from_pretrained(checkpoint_path).to(device)
    else:
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_path).to(device)
    model.eval()

    # --- Auto-detect trained image size from pos_embed (mirrors eval_layoutlm_all.py) ---
    # Hardcoding 512 caused "tensor a (197) must match tensor b (577)" when the
    # processor defaulted to 224px but the checkpoint was trained at 512px.
    img_size = 512  # fallback for non-LayoutLMv3 or legacy checkpoints
    if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "pos_embed"):
        n_tokens = model.layoutlmv3.pos_embed.shape[1]  # e.g. 1025 for 512px
        n_patches = n_tokens - 1                         # subtract CLS token
        patch_grid = int(round(n_patches ** 0.5))
        if patch_grid * patch_grid == n_patches:
            img_size = patch_grid * 16                   # patch_size=16 always

    processor.image_processor.size = {"height": img_size, "width": img_size}
    processor.image_processor.do_resize = True
    processor.image_processor.do_pad = True

    # Aspect-ratio-preserving resize + pad — must match training preprocessing.
    def _pad_to_square_infer(img, target=img_size):
        w, h = img.size
        scale = target / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        canvas = Image.new("RGB", (target, target), (255, 255, 255))
        canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((target - nw) // 2, (target - nh) // 2))
        return canvas

    image_for_model = _pad_to_square_infer(image)

    # 3. Predict Token Labels
    encoding = processor(
        image_for_model,
        words,
        boxes=boxes_1000,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    
    word_ids = encoding.word_ids(batch_index=0)
    inputs = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()
    
    # 4. Collapse Token Labels back down to Word sequence using word_ids
    # (Matches eval protocol: only take first sub-token predicting a whole word)
    word_preds = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word_id:
            word_preds.append(predictions[idx])
            previous_word_id = word_id

    # 5. Build BIO entities sequentially
    blocks = []
    current_tokens = []
    current_type = None
    current_boxes = []
    
    id2label = model.config.id2label

    for word, pred_id, px_box in zip(words, word_preds, boxes_px):
        label = id2label[int(pred_id)]
        
        if label.startswith("B-"):
            # Flush the current stack
            if current_tokens:
                blocks.append({
                    "id": len(blocks),
                    "label": current_type.lower() if current_type else "other",
                    "text": " ".join(current_tokens),
                    "bbox": _union_box_px(current_boxes),
                })
            current_tokens = [word]
            current_type = label[2:]  # "QUESTION", "HEADER" etc.
            current_boxes = [px_box]
            
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(word)
            current_boxes.append(px_box)
            
        else:
            # "O" tag or disjoint I-tag. Flush out current context
            if current_tokens:
                blocks.append({
                    "id": len(blocks),
                    "label": current_type.lower() if current_type else "other",
                    "text": " ".join(current_tokens),
                    "bbox": _union_box_px(current_boxes),
                })
            current_tokens = []
            current_type = None
            current_boxes = []
            
            if label == "O":
                blocks.append({
                    "id": len(blocks),
                    "label": "other",
                    "text": word,
                    "bbox": px_box,
                })

    # Flush dangling buffers
    if current_tokens:
        blocks.append({
            "id": len(blocks),
            "label": current_type.lower() if current_type else "other",
            "text": " ".join(current_tokens),
            "bbox": _union_box_px(current_boxes),
        })

    return {
        "blocks": blocks,
        "links": [],
        "tables": [],
        "confidence": sum(confidences)/len(confidences) if confidences else 1.0
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Custom OCR Training Pipeline")
    parser.add_argument("--ocr_engine", type=str, default="tesseract", choices=["tesseract", "paddle", "doctr"], help="OCR engine to use")
    parser.add_argument("--arch", type=str, default="layoutlmv3", choices=["layoutlmv3", "docformer"], help="Model backbone")
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"], help="LayoutLMv3 size")
    parser.add_argument("--speed_log_steps", type=int, default=50, help="Log per-step time every N steps")
    parser.add_argument("--speed_sync_mps", action="store_true", help="Synchronize MPS for accurate step timing")
    parser.add_argument("--speed_log_mps_mem", action="store_true", help="Log MPS memory usage with speed logs")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--cache_dir", type=str, required=True, help="Absolute path to HF cache directory")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Warm-start from an existing checkpoint (e.g., layoutlmv3-funsd/checkpoint-646)")
    parser.add_argument("--min_label_iou", type=float, default=_MIN_LABEL_IOU, help="Minimum IoU for assigning GT label to OCR box")
    parser.add_argument("--target_size", type=int, default=512, help="Square image size used for LayoutLM visual input (must be multiple of 16)")
    parser.add_argument("--output_suffix", type=str, default=None, help="Optional suffix appended to run tag to avoid output directory collisions")
    args = parser.parse_args()

    train(
        ocr_engine=args.ocr_engine,
        arch=args.arch,
        model_size=args.model_size,
        speed_log_steps=args.speed_log_steps,
        speed_sync_mps=args.speed_sync_mps,
        speed_log_mps_mem=args.speed_log_mps_mem,
        early_stopping_patience=args.early_stopping_patience,
        cache_dir=args.cache_dir,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        init_checkpoint=args.init_checkpoint,
        min_label_iou=args.min_label_iou,
        target_size=args.target_size,
        output_suffix=args.output_suffix,
    )
