"""
layoutlm_test.py
================
LayoutLMv3 inference & Q-A linking pipeline on FUNSD.

Loads a fine-tuned LayoutLMv3ForTokenClassification checkpoint, runs
word-level token classification on FUNSD test documents, converts the
BIO predictions into entity spans, and links QUESTION spans to the nearest
ANSWER span using a spatial proximity heuristic.

File location : LayoutLM/layoutlm_test.py
Checkpoint    : LayoutLM/layoutlmv3-funsd/checkpoint-<N>/
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)

# ============================================================
# Device
# ============================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Checkpoint
# ============================================================

CHECKPOINT = "./layoutlmv3-funsd/checkpoint-608"

# ============================================================
# Load Processor
# Trainer checkpoints don't save preprocessor_config.json.
# The image-processor config is fixed during fine-tuning —
# loading from the base model is correct and safe.
# ============================================================

processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)

# ============================================================
# Load Model
# Label mappings live in the checkpoint config — no need to
# pass num_labels / id2label / label2id again.
# ============================================================

model = LayoutLMv3ForTokenClassification.from_pretrained(CHECKPOINT)
model.to(device)
model.eval()

print("Loaded checkpoint:", model.config._name_or_path)

# ============================================================
# Label mappings  (derived from the checkpoint config)
# ============================================================

id2label   = model.config.id2label
label2id   = model.config.label2id
label_list = [id2label[i] for i in range(len(id2label))]

# ============================================================
# Load Dataset
# ============================================================

dataset = load_dataset("nielsr/funsd")

# ============================================================
# STEP 1 — BIO → Span Reconstruction
# ============================================================

def bio_to_spans(words, labels, bboxes):
    """
    Convert a BIO-tagged word sequence into a list of entity span dicts.

    Each span dict contains:
        type  : entity type string (e.g. "QUESTION", "ANSWER")
        text  : concatenated token text
        boxes : list of bounding boxes belonging to this span
    """
    spans = []
    current_tokens = []
    current_type   = None
    current_boxes  = []

    for word, label, box in zip(words, labels, bboxes):
        if label.startswith("B-"):
            # Flush the active span before starting a new one
            if current_tokens:
                spans.append({
                    "type":  current_type,
                    "text":  " ".join(current_tokens),
                    "boxes": current_boxes,
                })
            current_tokens = [word]
            current_type   = label[2:]
            current_boxes  = [box]

        elif label.startswith("I-") and current_type == label[2:]:
            # Continuation of the active span
            current_tokens.append(word)
            current_boxes.append(box)

        else:
            # "O" tag or mismatched I- tag — flush and reset
            if current_tokens:
                spans.append({
                    "type":  current_type,
                    "text":  " ".join(current_tokens),
                    "boxes": current_boxes,
                })
            current_tokens = []
            current_type   = None
            current_boxes  = []

    # Flush any remaining span
    if current_tokens:
        spans.append({
            "type":  current_type,
            "text":  " ".join(current_tokens),
            "boxes": current_boxes,
        })

    return spans


# ============================================================
# STEP 2 — Spatial Utility: Span Centre
# ============================================================

def span_center(boxes):
    """
    Return the (cx, cy) geometric centre of a set of bounding boxes.

    Args:
        boxes: list of [x1, y1, x2, y2] bounding boxes

    Returns:
        (cx, cy) tuple of floats
    """
    arr = np.array(boxes, dtype=float)   # shape (N, 4)
    x1  = arr[:, 0].mean()
    y1  = arr[:, 1].mean()
    x2  = arr[:, 2].mean()
    y2  = arr[:, 3].mean()
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# ============================================================
# STEP 3 — Spatial Q→A Linking
# ============================================================

def link_question_answers(spans):
    """
    Link each QUESTION span to its nearest ANSWER span using spatial proximity.

    Proximity score (lower = closer):
        score = vertical_distance + 0.5 * horizontal_distance

    Args:
        spans: list of span dicts produced by bio_to_spans()

    Returns:
        list of {"question": str, "answer": str} dicts
    """
    questions = [s for s in spans if s["type"] == "QUESTION"]
    answers   = [s for s in spans if s["type"] == "ANSWER"]

    linked = []

    for q in questions:
        qx, qy = span_center(q["boxes"])

        best_match    = None
        best_distance = float("inf")

        for a in answers:
            ax, ay = span_center(a["boxes"])

            vertical_distance   = abs(ay - qy)
            horizontal_distance = abs(ax - qx)

            # Combined proximity score — half-weight on horizontal drift
            distance = vertical_distance + 0.5 * horizontal_distance

            if distance < best_distance:
                best_distance = distance
                best_match    = a

        if best_match:
            linked.append({
                "question": q["text"],
                "answer":   best_match["text"],
            })

    return linked


# ============================================================
# STEP 4 — Full Inference Pipeline
# ============================================================

def predict_document(example):
    """
    Run the fine-tuned LayoutLMv3 model on a single raw FUNSD example
    and return a list of linked question-answer pairs.

    Pipeline:
        raw example → processor (real bboxes) → model logits
            → BIO label sequence → entity spans → Q-A pairs

    Args:
        example: one item from dataset["test"]
                 (must contain "image", "words", "bboxes")

    Returns:
        list of {"question": str, "answer": str} dicts
    """
    image  = example["image"].convert("RGB")
    words  = example["words"]
    bboxes = example["bboxes"]

    # Encode using real bounding boxes.
    # Keep as BatchEncoding (do NOT convert to plain dict yet) so that
    # word_ids() is still accessible for precise word alignment.
    encoding = processor(
        image,
        words,
        boxes=bboxes,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # word_ids() must be called before moving to device (plain dict loses it)
    word_ids = encoding.word_ids(batch_index=0)

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # predictions: (seq_len,)
    predictions = outputs.logits.argmax(-1).squeeze().cpu().numpy()

    # Align predictions to words using word_ids().
    # This mirrors exactly how seqeval evaluates during training:
    # only the first sub-token of each word is kept; CLS/SEP/PAD are skipped.
    word_level_preds = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word_id:
            word_level_preds.append(predictions[idx])
            previous_word_id = word_id

    labels = [label_list[int(p)] for p in word_level_preds]

    spans  = bio_to_spans(words, labels, bboxes)
    linked = link_question_answers(spans)

    return linked


# ============================================================
# STEP 5 — Debug Loop: First 5 Test Documents
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Inference: first 5 test documents")
    print("=" * 60)

    for doc_idx in range(5):
        example    = dataset["test"][doc_idx]
        doc_output = predict_document(example)

        print(f"\n--- Document {doc_idx} ---")
        if not doc_output:
            print("  (no Q-A pairs found)")
        else:
            for pair in doc_output:
                print(f"  Q: {pair['question']}")
                print(f"  A: {pair['answer']}")
                print()
