"""
layoutlm_visualize.py
=====================
LayoutLMv3 · Inference + Visualization Pipeline

Loads a trained LayoutLMv3ForTokenClassification checkpoint, runs inference
on FUNSD test samples, and saves colour-coded bounding-box overlay PNGs for
quick qualitative inspection.

No training logic.  No hard-coded local paths.
Works locally (MPS / CPU) and on Kaggle (P100/T4).

File location : LayoutLM/layoutlm_visualize.py
Project root  : formGeneration/

Structure
---------
1.  Imports
2.  Config
3.  Device setup
4.  Load model & processor
5.  Load dataset
6.  Utility functions
7.  Inference
8.  Visualization
9.  Debug runner
10. Main block
"""


# ============================================================
# 1.  IMPORTS
# ============================================================

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets import load_dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)


# ============================================================
# 2.  CONFIG  — tweak these without touching the logic below
# ============================================================

# Fine-tuned checkpoint directory.
# On Kaggle: use the path of your uploaded model dataset, e.g.
#   "/kaggle/input/layoutlmv3-funsd/checkpoint-608"
CHECKPOINT_PATH = "./layoutlmv3-funsd/checkpoint-608"

# Output directory for saved PNGs
# Using ../output_predictions so outputs land in the project-root directory
# (formGeneration/output_predictions/) rather than inside LayoutLM/.
OUTPUT_DIR = "../output_predictions"

# Number of test samples to debug
DEBUG_N = 5

# Whether to overlay ground-truth boxes alongside predictions
SHOW_GT = True

# Colour scheme — RGB tuples
LABEL_COLORS = {
    "QUESTION": (70,  130, 255),  # strong blue
    "ANSWER":   (50,  200, 100),  # strong green
    "HEADER":   (220,  60,  60),  # strong red
    "O":        None,             # transparent — skip
    # BIO prefix variants
    "B-QUESTION": (70,  130, 255),
    "I-QUESTION": (70,  130, 255),
    "B-ANSWER":   (50,  200, 100),
    "I-ANSWER":   (50,  200, 100),
    "B-HEADER":   (220,  60,  60),
    "I-HEADER":   (220,  60,  60),
}

# Lighter versions for ground-truth overlay (alpha blend towards white)
GT_BLEND = 0.45  # 0 = white, 1 = full colour

def _lighten(color: tuple[int, int, int], blend: float) -> tuple[int, int, int]:
    return tuple(int(255 * (1 - blend) + c * blend) for c in color)

LABEL_COLORS_GT = {
    k: (_lighten(v, GT_BLEND) if v is not None else None)
    for k, v in LABEL_COLORS.items()
}


# ============================================================
# 3.  DEVICE SETUP
# ============================================================

def get_device() -> torch.device:
    """Return MPS if available (Apple Silicon), else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
print(f"[device] Using: {device}")


# ============================================================
# 4.  LOAD MODEL & PROCESSOR
# ============================================================

def load_model_and_processor(
    checkpoint: str,
) -> tuple[LayoutLMv3Processor, LayoutLMv3ForTokenClassification, list[str]]:
    """
    Load LayoutLMv3Processor (from base model) and
    LayoutLMv3ForTokenClassification (from fine-tuned checkpoint).

    The processor is always loaded from 'microsoft/layoutlmv3-base' because
    HuggingFace Trainer checkpoints do not save preprocessor_config.json —
    the image-processor config is identical to the base model and never
    changes during fine-tuning.

    Label mappings are read directly from the checkpoint config so they are
    guaranteed to match the trained classifier head.
    """
    base = "microsoft/layoutlmv3-base"

    print(f"[model] Loading processor from: {base}")
    processor = LayoutLMv3Processor.from_pretrained(base, apply_ocr=False)

    print(f"[model] Loading model from:     {checkpoint}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint)

    model.to(device)
    model.eval()

    # Derive label list from checkpoint config — guaranteed consistent
    id2label   = model.config.id2label
    label_list = [id2label[i] for i in range(len(id2label))]

    print(f"Loaded checkpoint: {model.config._name_or_path}")
    print(f"[model] Labels ({len(label_list)}): {label_list}")

    return processor, model, label_list


# ============================================================
# 5.  LOAD DATASET
# ============================================================

def load_funsd() -> tuple:
    """
    Load the FUNSD dataset from HuggingFace Hub.
    Returns (dataset, label_list, id2label, label2id).
    """
    print("[data] Loading FUNSD dataset …")
    dataset = load_dataset("nielsr/funsd")
    label_list: list[str] = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: lbl for i, lbl in enumerate(label_list)}
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    print(f"[data] Labels: {label_list}")
    return dataset, label_list, id2label, label2id


# ============================================================
# 6.  UTILITY FUNCTIONS
# ============================================================

def unnormalize_box(
    bbox: list[int],
    width: int,
    height: int,
) -> list[float]:
    """
    Convert FUNSD-normalised box coordinates (0–1000 scale) back to
    pixel coordinates for a given image (width, height).
    """
    return [
        width  * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width  * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def label_to_canonical(label: str) -> str:
    """
    Strip BIO prefix.  'B-ANSWER' → 'ANSWER', 'O' → 'O'.
    """
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label


def get_color(label: str, gt: bool = False) -> tuple[int, int, int] | None:
    """Return draw colour for a label, or None to skip drawing."""
    palette = LABEL_COLORS_GT if gt else LABEL_COLORS
    # Try full label first, then canonical form
    color = palette.get(label) or palette.get(label_to_canonical(label))
    return color


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# 7.  INFERENCE
# ============================================================

def predict_tokens(
    example: dict,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForTokenClassification,
    label_list: list[str],
) -> list[str]:
    """
    Run LayoutLMv3 inference on a single raw FUNSD example.

    Args:
        example:    One item from dataset["test"] containing
                    "image", "words", and "bboxes".
        processor:  Loaded LayoutLMv3Processor.
        model:      Loaded LayoutLMv3ForTokenClassification (eval mode).
        label_list: Ordered list of label strings.

    Returns:
        List of predicted label strings aligned to the original words list.
        Length == len(example["words"]).
    """
    image = example["image"].convert("RGB")
    words = example["words"]
    bboxes = example["bboxes"]

    # Encode with real bounding boxes.
    # Keep as BatchEncoding (do NOT convert to plain dict yet) so that
    # word_ids() is still accessible for exact word alignment.
    encoding = processor(
        image,
        words,
        boxes=bboxes,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # Extract word_ids BEFORE moving to device — plain dict loses this method.
    word_ids = encoding.word_ids(batch_index=0)

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # predictions: (seq_len,)
    predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()

    # Align predictions to words using word_ids().
    # Mirrors exactly how seqeval evaluates during training:
    # only the first sub-token of each word is kept; CLS/SEP/PAD are skipped.
    word_level_preds = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word_id:
            word_level_preds.append(predictions[idx])
            previous_word_id = word_id

    # Guard against truncation
    if len(word_level_preds) < len(words):
        o_id = label_list.index("O") if "O" in label_list else 0
        word_level_preds += [o_id] * (len(words) - len(word_level_preds))

    return [label_list[int(p)] for p in word_level_preds]


# ============================================================
# 8.  VISUALIZATION
# ============================================================

def visualize_predictions(
    example: dict,
    predicted_labels: list[str],
    label_list: list[str],
    show_gt: bool = SHOW_GT,
) -> Image.Image:
    """
    Draw coloured bounding-box overlays on the document image.

    Colour scheme (strong):
        QUESTION → blue
        ANSWER   → green
        HEADER   → red
        O        → skipped (transparent)

    When show_gt is True, ground-truth boxes are drawn first in lighter
    colours, then prediction boxes are drawn on top in strong colours.
    This allows easy visual comparison of GT vs. predicted labels.

    Args:
        example:          Raw FUNSD test example.
        predicted_labels: Word-level predicted label strings.
        label_list:       Full label list for reference.
        show_gt:          Whether to overlay ground-truth boxes.

    Returns:
        Annotated PIL Image.
    """
    image = example["image"].convert("RGB")
    width, height = image.size

    words  = example["words"]
    bboxes = example["bboxes"]
    gt_labels = [label_list[t] for t in example["ner_tags"]]

    draw = ImageDraw.Draw(image, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    # — Ground-truth boxes (lighter colours, drawn first / behind) —
    if show_gt:
        for bbox, gt_lbl in zip(bboxes, gt_labels):
            color = get_color(gt_lbl, gt=True)
            if color is None:
                continue
            px_box = unnormalize_box(bbox, width, height)
            draw.rectangle(px_box, outline=color + (180,), width=2)

    # — Prediction boxes (strong colours, drawn on top) —
    for bbox, pred_lbl in zip(bboxes, predicted_labels):
        color = get_color(pred_lbl, gt=False)
        if color is None:
            continue
        px_box = unnormalize_box(bbox, width, height)
        draw.rectangle(px_box, outline=color + (255,), width=2)

    return image


def make_side_by_side(
    example: dict,
    predicted_labels: list[str],
    label_list: list[str],
) -> Image.Image:
    """
    Create a side-by-side comparison image:
        Left  → Ground truth boxes (strong colours)
        Right → Prediction boxes  (strong colours)

    This variant draws each panel independently for the clearest comparison.
    """
    def _draw_boxes(example, labels_str, lighter=False):
        img   = example["image"].convert("RGB")
        w, h  = img.size
        draw  = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            font = ImageFont.load_default()

        for bbox, lbl in zip(example["bboxes"], labels_str):
            color = get_color(lbl, gt=lighter)
            if color is None:
                continue
            px_box = unnormalize_box(bbox, w, h)
            draw.rectangle(px_box, outline=color + (255,), width=2)
        return img

    gt_labels_str   = [label_list[t] for t in example["ner_tags"]]
    gt_image        = _draw_boxes(example, gt_labels_str,    lighter=False)
    pred_image      = _draw_boxes(example, predicted_labels, lighter=False)

    # Combine horizontally
    total_w = gt_image.width + pred_image.width + 10   # 10 px gap
    combined = Image.new("RGB", (total_w, max(gt_image.height, pred_image.height)), (255, 255, 255))
    combined.paste(gt_image,   (0, 0))
    combined.paste(pred_image, (gt_image.width + 10, 0))

    # Add simple header text
    draw = ImageDraw.Draw(combined)
    try:
        font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font_hdr = ImageFont.load_default()

    draw.text((10, 5),                      "Ground Truth", fill=(30, 30, 30), font=font_hdr)
    draw.text((gt_image.width + 20, 5),     "Prediction",   fill=(30, 30, 30), font=font_hdr)

    return combined


# ============================================================
# 9.  BOX STATISTICS
# ============================================================

def print_box_stats(
    example: dict,
    predicted_labels: list[str],
    label_list: list[str],
    doc_id: str = "",
) -> None:
    """
    Print a per-type breakdown of ground-truth vs predicted box counts,
    and report how many non-O boxes were missed (predicted as O) and
    how many O boxes were falsely detected as a named entity.

    Args:
        example:          Raw FUNSD example (must contain "ner_tags", "words").
        predicted_labels: Word-level predicted label strings (from predict_tokens).
        label_list:       Full ordered label list.
        doc_id:           Optional identifier shown in the header (e.g. doc index).
    """
    from collections import Counter

    gt_labels  = [label_list[t] for t in example["ner_tags"]]
    pred_labels = predicted_labels          # already word-aligned strings

    # Canonical (strip BIO prefix) counts
    def canonical(lbl):
        return lbl[2:] if lbl.startswith(("B-", "I-")) else lbl

    gt_count   = Counter(canonical(l) for l in gt_labels)
    pred_count = Counter(canonical(l) for l in pred_labels)

    # All entity types (excluding O)
    entity_types = sorted(set(gt_count) | set(pred_count) - {"O"})

    header = f"  Doc {doc_id}" if doc_id != "" else "  Document"
    print(f"\n{'─' * 52}")
    print(f"{header} — box statistics")
    print(f"{'─' * 52}")
    print(f"  {'Type':<12}  {'GT':>6}  {'Pred':>6}  {'Δ':>6}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}")
    for typ in entity_types + ["O"]:
        gt_n   = gt_count.get(typ, 0)
        pred_n = pred_count.get(typ, 0)
        delta  = pred_n - gt_n
        sign   = "+" if delta > 0 else ""
        print(f"  {typ:<12}  {gt_n:>6}  {pred_n:>6}  {sign}{delta:>5}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}")
    total_gt   = sum(gt_count.values())
    total_pred = sum(pred_count.values())
    print(f"  {'TOTAL':<12}  {total_gt:>6}  {total_pred:>6}")

    # Missed: GT was a named entity but pred is O
    missed  = sum(
        1 for gt, pr in zip(gt_labels, pred_labels)
        if canonical(gt) != "O" and canonical(pr) == "O"
    )
    # False detections: GT is O but pred is a named entity
    false_det = sum(
        1 for gt, pr in zip(gt_labels, pred_labels)
        if canonical(gt) == "O" and canonical(pr) != "O"
    )
    print(f"\n  Missed (GT≠O → pred=O)  : {missed}")
    print(f"  False  (GT=O  → pred≠O) : {false_det}")
    print(f"{'─' * 52}\n")


# ============================================================
# 10.  DEBUG RUNNER
# ============================================================

def debug_first_n(
    n: int,
    dataset,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForTokenClassification,
    label_list: list[str],
    output_dir: str = OUTPUT_DIR,
    show_gt: bool = SHOW_GT,
    save_comparison: bool = True,
) -> None:
    """
    Run inference on the first *n* test samples and save output PNGs.
    After each document, prints a box-count statistics table showing
    GT vs predicted counts per label type, missed boxes, and false detections.

    Outputs
    -------
    For each document i:
        {output_dir}/pred_doc_{i}.png         — prediction overlay
        {output_dir}/comparison_doc_{i}.png   — side-by-side GT vs. Pred
                                                 (when save_comparison=True)
    """
    ensure_output_dir(output_dir)
    test_split = dataset["test"]

    for i in range(min(n, len(test_split))):
        example = test_split[i]
        print(f"[debug] Processing document {i} …", end="  ")

        predicted_labels = predict_tokens(example, processor, model, label_list)

        # — Prediction overlay —
        overlay = visualize_predictions(example, predicted_labels, label_list, show_gt=show_gt)
        pred_path = os.path.join(output_dir, f"pred_doc_{i}.png")
        overlay.save(pred_path)
        print(f"saved → {pred_path}", end="")

        # — Side-by-side comparison —
        if save_comparison:
            comparison = make_side_by_side(example, predicted_labels, label_list)
            cmp_path = os.path.join(output_dir, f"comparison_doc_{i}.png")
            comparison.save(cmp_path)
            print(f"  |  {cmp_path}", end="")

        print()   # newline

        # — Per-document box statistics —
        print_box_stats(example, predicted_labels, label_list, doc_id=i)

    print(f"[debug] Done.  All outputs saved to: {output_dir}/")


# ============================================================
# 9b.  RUN ON SPECIFIC DOCUMENT BY FILENAME
# ============================================================

def run_on_doc(
    doc_name: str,
    dataset,
    processor,
    model,
    label_list: list[str],
    split: str = "test",
    output_dir: str = OUTPUT_DIR,
    show_gt: bool = SHOW_GT,
) -> str:
    """
    Run inference on a single FUNSD document identified by its filename stem.

    Args:
        doc_name : filename with or without extension,
                   e.g. '0011505151' or '0011505151.png'
        split    : dataset split to search — 'test' or 'train'

    Returns:
        Path of the saved comparison PNG.

    Raises:
        ValueError if the document is not found in the split.
    """
    stem = os.path.splitext(doc_name)[0]   # strip .png / .jpg if present

    # nielsr/funsd 'id' is a sequential integer ("0", "1", …), NOT the original
    # FUNSD filename.  The true filename lives in the raw Arrow image column
    # as image_dict["path"].  Access it without decoding the full image bytes.
    example = None
    image_col = dataset[split].data.column("image")
    for i, cell in enumerate(image_col):
        img_path = cell.as_py().get("path", "")
        path_stem = os.path.splitext(img_path)[0]
        if path_stem == stem:
            example = dataset[split][i]
            break

    if example is None:
        raise ValueError(
            f"Document '{stem}' not found in dataset['{split}']. "
            f"Check spelling or try the other split."
        )

    print(f"[run_on_doc] Found '{stem}' in split='{split}'")

    predicted_labels = predict_tokens(example, processor, model, label_list)

    ensure_output_dir(output_dir)

    overlay = visualize_predictions(example, predicted_labels, label_list, show_gt=show_gt)
    pred_path = os.path.join(output_dir, f"pred_{stem}.png")
    overlay.save(pred_path)
    print(f"[run_on_doc] Prediction overlay → {pred_path}")

    comparison = make_side_by_side(example, predicted_labels, label_list)
    cmp_path = os.path.join(output_dir, f"comparison_{stem}.png")
    comparison.save(cmp_path)
    print(f"[run_on_doc] Comparison image   → {cmp_path}")

    # — Box statistics for this document —
    print_box_stats(example, predicted_labels, label_list, doc_id=stem)

    return cmp_path


# ============================================================
# 10. MAIN BLOCK
# ============================================================

if __name__ == "__main__":
    # --- Load model & processor (label_list comes from checkpoint config) ---
    processor, model, label_list = load_model_and_processor(CHECKPOINT_PATH)

    # --- Load dataset ---
    dataset, _, _, _ = load_funsd()

    # --- Run on a specific document by filename ---
    # Change doc_name to any FUNSD image filename stem.
    # Use split="train" if not found in test.
    run_on_doc(
        doc_name="00836244",
        dataset=dataset,
        processor=processor,
        model=model,
        label_list=label_list,
        split="train",          # 0011505151 lives in the train split
        output_dir=OUTPUT_DIR,
        show_gt=SHOW_GT,
    )

    # --- Batch debug (uncomment to run first N docs) ---
    # debug_first_n(
    #     n=DEBUG_N,
    #     dataset=dataset,
    #     processor=processor,
    #     model=model,
    #     label_list=label_list,
    #     output_dir=OUTPUT_DIR,
    #     show_gt=SHOW_GT,
    #     save_comparison=True,
    # )
