"""
donut_model.py
==============
Minimal Donut (vision-encoder/decoder) experiment on FUNSD.

The pipeline differs from token-classification models:
- Convert BIO tags into a text target: <question> ... <answer> ...
- Train Donut (image -> JSON-like text) with seq2seq training.
- Evaluate by parsing generated text back into entities and computing F1.

File location : Donut/donut_model.py
Output        : Donut/donut-funsd/
"""

import re
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# =========================
# Device
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# =========================
# Helpers: BIO -> spans -> paired text
# =========================
def bio_to_spans(words: List[str], label_ids: List[int], label_list: List[str]) -> List[Tuple[str, str]]:
    """Convert BIO tags to (entity_type, text) spans in reading order."""
    spans: List[Tuple[str, str]] = []
    current_tokens: List[str] = []
    current_type: str | None = None

    for word, tag_id in zip(words, label_ids):
        label = label_list[tag_id]

        if label.startswith("B-"):
            if current_tokens:
                spans.append((current_type, " ".join(current_tokens)))
            current_tokens = [word]
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(word)
        else:
            if current_tokens:
                spans.append((current_type, " ".join(current_tokens)))
            current_tokens = []
            current_type = None

    if current_tokens:
        spans.append((current_type, " ".join(current_tokens)))

    return spans


def spans_to_target(spans: List[Tuple[str, str]]) -> str:
    """Pair QUESTION and ANSWER spans and flatten to Donut target string."""
    questions = [text for ent, text in spans if ent == "QUESTION"]
    answers = [text for ent, text in spans if ent == "ANSWER"]

    pairs: List[Tuple[str, str]] = list(zip(questions, answers))

    # Handle unpaired leftovers gracefully
    if not pairs and (questions or answers):
        filler = ""  # keep empty counterpart when unmatched
        if questions:
            pairs = [(q, answers[i] if i < len(answers) else filler) for i, q in enumerate(questions)]
        else:
            pairs = [(filler, a) for a in answers]

    pieces = [f"<question> {q.strip()} <answer> {a.strip()}" for q, a in pairs if q or a]
    return " ".join(pieces) if pieces else "<question> <answer>"


# =========================
# Dataset & processor
# =========================
dataset = load_dataset("nielsr/funsd")
label_list = dataset["train"].features["ner_tags"].feature.names

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Configure special tokens for training/generation
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.tie_word_embeddings = False  # silence tie-weight warning from checkpoint
max_length = 256
generation_max_length = 256


# =========================
# Preprocessing
# =========================
def preprocess(batch):
    # with_transform passes a dict of lists (one entry per column per sample)
    all_pixel_values = []
    all_labels = []
    all_target_texts = []

    for image, words, ner_tags in zip(batch["image"], batch["words"], batch["ner_tags"]):
        image = image.convert("RGB")
        spans = bio_to_spans(words, ner_tags, label_list)
        target_text = spans_to_target(spans)

        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze(0)

        label_ids = processor.tokenizer(
            target_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        label_ids[label_ids == processor.tokenizer.pad_token_id] = -100

        all_pixel_values.append(pixel_values)
        all_labels.append(label_ids)
        all_target_texts.append(target_text)

    return {
        "pixel_values": all_pixel_values,   # list of tensors
        "labels": all_labels,               # list of tensors
        "target_text": all_target_texts,    # list of strings
    }


# On-the-fly transforms to avoid Arrow overflow when storing large tensors
train_dataset = dataset["train"].with_transform(preprocess)
eval_dataset = dataset["test"].with_transform(preprocess)


# =========================
# Collator
# =========================
def collate_fn(batch):
    # batch is a list of dicts; each value may already be a tensor (from batched transform)
    pixel_values = torch.stack([
        item["pixel_values"] if isinstance(item["pixel_values"], torch.Tensor)
        else torch.tensor(item["pixel_values"])
        for item in batch
    ])
    labels = torch.stack([
        item["labels"] if isinstance(item["labels"], torch.Tensor)
        else torch.tensor(item["labels"])
        for item in batch
    ])
    return {"pixel_values": pixel_values, "labels": labels}


# =========================
# Metrics
# =========================
def parse_pairs(text: str) -> List[Tuple[str, str]]:
    """Extract (question, answer) pairs from generated text."""
    # Remove special tokens that sometimes linger after decoding
    cleaned = text.replace(processor.tokenizer.eos_token or "", "").strip()
    pattern = re.compile(r"<question>\s*(.*?)\s*<answer>\s*(.*?)(?=<question>|$)", re.IGNORECASE | re.DOTALL)
    pairs: List[Tuple[str, str]] = []
    for match in pattern.finditer(cleaned):
        q = match.group(1).strip()
        a = match.group(2).strip()
        if q or a:
            pairs.append((q.lower(), a.lower()))
    return pairs


def f1_from_pairs(pred_pairs: List[List[Tuple[str, str]]], label_pairs: List[List[Tuple[str, str]]]):
    tp = fp = fn = 0
    for preds, labels in zip(pred_pairs, label_pairs):
        pred_set = set(preds)
        label_set = set(labels)
        tp += len(pred_set & label_set)
        fp += len(pred_set - label_set)
        fn += len(label_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Seq2SeqTrainer may return a tuple (generated tokens, ...)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    pred_pairs = [parse_pairs(p) for p in decoded_preds]
    label_pairs = [parse_pairs(l) for l in decoded_labels]

    return f1_from_pairs(pred_pairs, label_pairs)


# =========================
# Training
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-funsd",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # effective batch size = 4
    num_train_epochs=5,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=False,
    generation_max_length=max_length,
    remove_unused_columns=False,  # we only feed pixel_values/labels
    dataloader_pin_memory=False,
)

trainer = Seq2SeqTrainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)


if __name__ == "__main__":
    trainer.train()
