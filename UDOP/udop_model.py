"""
udop_model.py
=============
UDOP token classification on FUNSD.

UdopForTokenClassification does not exist in the transformers library.
We use UdopEncoderModel (encoder-only, T5-style) and attach a linear
classification head — this gives a fair apples-to-apples comparison
against LayoutLMv3 (both are encoder-only for this task).

Model:  microsoft/udop-large  (only published UDOP checkpoint)
        d_model = 1024, so the head is Linear(1024, num_labels).

File location : UDOP/udop_model.py
Output        : UDOP/udop-funsd/
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    UdopProcessor,
    TrainingArguments,
    Trainer,
)
from transformers import UdopForConditionalGeneration
from transformers.modeling_outputs import TokenClassifierOutput
import evaluate


# =========================
# Device
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# =========================
# Dataset
# =========================
dataset = load_dataset("nielsr/funsd")

# Collect all training labels
all_labels = []
for example in dataset["train"]:
    all_labels.extend(example["ner_tags"])

classes = np.unique(all_labels)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=all_labels
)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)


# =========================
# Custom model: UDOP encoder + token-classification head
# =========================

class UdopForTokenClassificationCustom(nn.Module):
    def __init__(self, model_name, num_labels, class_weights):
        super().__init__()
        self.udop = UdopForConditionalGeneration.from_pretrained(model_name)
        hidden_size = self.udop.config.d_model
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Register as buffer so it moves with .to(device) automatically,
        # avoiding device-mismatch errors during eval or multi-GPU.
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        bbox=None,
        pixel_values=None,
        labels=None,
        **kwargs,   # absorb any extra fields the Trainer may pass
    ):
        outputs = self.udop.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        # UDOP encoder appends visual patch tokens after the text tokens,
        # so last_hidden_state is (B, text_len + visual_patches, d_model).
        # Labels only cover the text portion — slice to match.
        text_len = input_ids.shape[1]
        sequence_output = sequence_output[:, :text_len, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Build loss with the registered buffer (stays on the correct device).
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=-100,
            )
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(loss=loss, logits=logits)


# =========================
# Processor & Model
# =========================
MODEL_ID = "microsoft/udop-large"
processor = UdopProcessor.from_pretrained(MODEL_ID, use_fast=False, apply_ocr=False)

# The custom model loads pretrained weights internally via from_pretrained,
# so no separate base_encoder download or weight copy is needed.
model = UdopForTokenClassificationCustom(
    MODEL_ID,
    num_labels=len(label_list),
    class_weights=class_weights
).to(device)

# model.udop is UdopForConditionalGeneration, which has a .config attribute.
model.udop.config.id2label = id2label
model.udop.config.label2id = label2id


# =========================
# Encoding
# =========================
def encode(example):
    image = example["image"].convert("RGB")

    encoding = processor(
        images=image,
        text=example["words"],
        boxes=example["bboxes"],
        word_labels=example["ner_tags"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    encoding = {k: v.squeeze() for k, v in encoding.items()}
    return encoding


encoded_dataset = dataset.map(
    encode,
    remove_columns=dataset["train"].column_names,
)


# =========================
# Metrics
# =========================
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_preds = [
        [label_list[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_preds, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# =========================
# Training Args
# =========================
# Total steps = ceil(149 / 2) * 15 = 75 * 15 = 1125  →  10% warmup ≈ 113 steps
training_args = TrainingArguments(
    output_dir="./udop-funsd",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    learning_rate=1.5e-5,
    warmup_steps=113,        # warmup_ratio deprecated in transformers >=5.2
    weight_decay=0.01,
    eval_strategy="epoch",   # renamed from evaluation_strategy in transformers >=4.41
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    dataloader_pin_memory=False,  # required on MPS
    save_safetensors=False,       # UDOP ties embedding weights; safetensors rejects shared tensors
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=processor,   # 'processing_class' was renamed from 'tokenizer' only in transformers >=5.x
    compute_metrics=compute_metrics,
)

trainer.train()
