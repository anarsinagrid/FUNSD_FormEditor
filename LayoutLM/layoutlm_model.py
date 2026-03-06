"""
layoutlm_model.py
=================
LayoutLMv3 fine-tuning on FUNSD .

Trains LayoutLMv3ForTokenClassification from the microsoft/layoutlmv3-base
checkpoint using the FUNSD named-entity recognition dataset.  

Uses a layer-wise learning-rate decay (LLRD) optimizer so that lower
transformer layers receive a smaller learning rate than the classifier head.

File location : LayoutLM/layoutlm_model.py
Output        : LayoutLM/layoutlmv3-funsd-zero-bbox/
"""

import torch
from datasets import load_dataset
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer
)
from torch.optim import AdamW
import evaluate
import numpy as np

# =========================
# Device
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================
# Load Dataset
# =========================
dataset = load_dataset("nielsr/funsd")

# =========================
# Processor
# =========================
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)

# NOTE: Do NOT change image_processor.size — LayoutLMv3-base's ViT backbone
# has fixed positional embeddings for 224x224 (197 positions).
# Changing the resolution causes a tensor size mismatch at runtime.

# =========================
# Label mappings
# =========================
label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# =========================
# Encoding
# =========================
def encode_zero_bbox(example):
    image = example["image"].convert("RGB")

    # All bounding boxes zeroed out — ablation to measure spatial info contribution.
    zero_boxes = [[0, 0, 0, 0] for _ in example["bboxes"]]

    encoding = processor(
        image,
        example["words"],
        boxes=zero_boxes,
        word_labels=example["ner_tags"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    return {k: v.squeeze() for k, v in encoding.items()}

encoded_dataset = dataset.map(
    encode_zero_bbox,
    remove_columns=dataset["train"].column_names
)

# =========================
# Model
# =========================
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

model.to(device)

# =========================
# Metrics
# =========================
metric = evaluate.load("seqeval")

def compute_metrics(p):
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

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# =========================
# Layer-wise LR Decay Optimizer
# =========================
def get_layerwise_lr_decay_optimizer(model, base_lr=2e-5, layer_decay=0.95, weight_decay=0.01):
    lm = model.layoutlmv3
    num_layers = len(lm.encoder.layer)
    optimizer_grouped_parameters = []

    # Text + layout embeddings (lowest LR)
    lr = base_lr * (layer_decay ** (num_layers + 1))
    optimizer_grouped_parameters.append({
        "params": lm.embeddings.parameters(),
        "lr": lr,
        "weight_decay": weight_decay
    })

    # Visual backbone patch embedding — was previously uncovered and fell back to
    # PyTorch default LR (1e-3). Assign it the same low LR as text embeddings.
    if hasattr(lm, "patch_embed"):
        optimizer_grouped_parameters.append({
            "params": lm.patch_embed.parameters(),
            "lr": lr,
            "weight_decay": weight_decay
        })

    # Transformer layers (increasing LR toward the top)
    for layer_idx in range(num_layers):
        layer = lm.encoder.layer[layer_idx]
        lr = base_lr * (layer_decay ** (num_layers - layer_idx))
        optimizer_grouped_parameters.append({
            "params": layer.parameters(),
            "lr": lr,
            "weight_decay": weight_decay
        })

    # Classifier head (full base LR)
    optimizer_grouped_parameters.append({
        "params": model.classifier.parameters(),
        "lr": base_lr,
        "weight_decay": weight_decay
    })

    return AdamW(optimizer_grouped_parameters)

optimizer = get_layerwise_lr_decay_optimizer(
    model,
    base_lr=2e-5,
    layer_decay=0.95,
    weight_decay=0.01
)

# =========================
# Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir="./layoutlmv3-funsd-zero-bbox",  # distinct from baseline run
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30, 
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,  # gradient clipping
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=processor,       # use 'processing_class' if transformers >=4.46
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)

# =========================
# Train
# =========================
trainer.train()