"""
layout_graph_pipeline.py
========================
Standalone graph-enhanced LayoutLMv3 experimentation pipeline (FUNSD + docTR).

This script is intentionally isolated from LayoutLM/graph_linking and editor
runtime defaults.

Subcommands:
  - train
  - evaluate
  - relation_eval
  - importance
  - run_all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import evaluate as hf_evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

try:
    from torch_geometric.nn import GATv2Conv

    HAS_PYG = True
except Exception:
    GATv2Conv = None
    HAS_PYG = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from layoutlm_customOCR import OCRBackend, align_ocr_to_ground_truth

DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "layoutlmv3-funsd-doctr", "checkpoint-800")
DEFAULT_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, ".hf_cache")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "layoutlmv3-funsd-doctr-layoutgraph")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "layout_graph_artifacts")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "layout_graph_report.json")

RELATION_NAME_TO_ID = {
    "above": 0,
    "below": 1,
    "left-of": 2,
    "right-of": 3,
    "aligned-with": 4,
}
RELATION_ID_TO_NAME = {v: k for k, v in RELATION_NAME_TO_ID.items()}


@dataclass
class PreparedSample:
    inputs: Dict[str, torch.Tensor]
    token_labels: torch.Tensor
    word_ids: List[Optional[int]]
    boxes_1000: List[List[int]]
    aligned_word_label_ids: List[int]
    relation_pairs: List[Tuple[int, int, float]]
    low_conf_ratio: float


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _require_pyg() -> None:
    if HAS_PYG:
        return
    raise RuntimeError(
        "torch_geometric is required for layout_graph_pipeline. "
        "Install dependencies from requirements.txt (including torch-geometric)."
    )


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _bbox_center(box: Sequence[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def _bbox_union(boxes: List[Sequence[int]]) -> List[int]:
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(int(b[0]) for b in boxes)
    y1 = min(int(b[1]) for b in boxes)
    x2 = max(int(b[2]) for b in boxes)
    y2 = max(int(b[3]) for b in boxes)
    return [x1, y1, x2, y2]


def _max_word_id(word_ids: Sequence[Optional[int]]) -> int:
    max_wid = -1
    for wid in word_ids:
        if wid is not None and wid > max_wid:
            max_wid = wid
    return max_wid


# ---------------------------------------------------------------------------
# Relation / graph feature helpers
# ---------------------------------------------------------------------------

def _relation_id_from_boxes(box_i: Sequence[int], box_j: Sequence[int]) -> int:
    cx_i, cy_i = _bbox_center(box_i)
    cx_j, cy_j = _bbox_center(box_j)
    dx = cx_j - cx_i
    dy = cy_j - cy_i

    if abs(cx_i - cx_j) <= 20.0 or abs(cy_i - cy_j) <= 20.0:
        return RELATION_NAME_TO_ID["aligned-with"]
    if abs(dx) >= abs(dy):
        return RELATION_NAME_TO_ID["right-of"] if dx >= 0 else RELATION_NAME_TO_ID["left-of"]
    return RELATION_NAME_TO_ID["below"] if dy >= 0 else RELATION_NAME_TO_ID["above"]


def _overlap_ratio_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, max(a1 - a0, b1 - b0))
    return float(inter / denom)


def _edge_cont_features(box_i: Sequence[int], box_j: Sequence[int]) -> List[float]:
    cx_i, cy_i = _bbox_center(box_i)
    cx_j, cy_j = _bbox_center(box_j)
    dx = (cx_j - cx_i) / 1000.0
    dy = (cy_j - cy_i) / 1000.0
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    dist = math.sqrt(dx * dx + dy * dy)

    x1_i, y1_i, x2_i, y2_i = [float(v) for v in box_i]
    x1_j, y1_j, x2_j, y2_j = [float(v) for v in box_j]

    y_overlap = _overlap_ratio_1d(y1_i, y2_i, y1_j, y2_j)
    x_overlap = _overlap_ratio_1d(x1_i, x2_i, x1_j, x2_j)

    return [
        float(dx),
        float(dy),
        float(abs_dx),
        float(abs_dy),
        float(dist),
        float(y_overlap),
        float(x_overlap),
    ]


def _build_knn_graph(
    boxes_1000: List[List[int]],
    neighbor_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(boxes_1000)
    if n <= 1:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0, 7), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    k = max(1, min(int(neighbor_k), max(1, n - 1)))

    centers = np.array([_bbox_center(b) for b in boxes_1000], dtype=np.float32)
    src: List[int] = []
    dst: List[int] = []
    cont_feats: List[List[float]] = []
    rel_ids: List[int] = []

    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            d = float(dx * dx + dy * dy)
            dists.append((d, j))

        dists.sort(key=lambda x: x[0])
        for _, j in dists[:k]:
            src.append(i)
            dst.append(j)
            cont_feats.append(_edge_cont_features(boxes_1000[i], boxes_1000[j]))
            rel_ids.append(_relation_id_from_boxes(boxes_1000[i], boxes_1000[j]))

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(cont_feats, dtype=torch.float32, device=device)
    edge_rel_ids = torch.tensor(rel_ids, dtype=torch.long, device=device)
    return edge_index, edge_attr, edge_rel_ids


# ---------------------------------------------------------------------------
# Entity + pseudo related-field targets
# ---------------------------------------------------------------------------

def _label_to_entity_type(label_name: str) -> Optional[str]:
    if label_name.startswith("B-") or label_name.startswith("I-"):
        t = label_name.split("-", 1)[1].lower()
        if t in {"question", "answer", "header"}:
            return t
    return None


def _extract_entities_from_word_labels(
    word_label_ids: Sequence[int],
    id2label: Dict[int, str],
    boxes_1000: Sequence[Sequence[int]],
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []

    current_type: Optional[str] = None
    current_word_indices: List[int] = []

    def flush() -> None:
        nonlocal current_type, current_word_indices
        if not current_word_indices or current_type is None:
            current_word_indices = []
            current_type = None
            return
        ent_boxes = [boxes_1000[w] for w in current_word_indices if 0 <= w < len(boxes_1000)]
        if not ent_boxes:
            current_word_indices = []
            current_type = None
            return
        union_box = _bbox_union(ent_boxes)
        cx, cy = _bbox_center(union_box)
        entities.append(
            {
                "type": current_type,
                "word_indices": list(current_word_indices),
                "first_word": int(current_word_indices[0]),
                "box": union_box,
                "center": (cx, cy),
            }
        )
        current_word_indices = []
        current_type = None

    for i, lid in enumerate(word_label_ids):
        label_name = id2label.get(int(lid), "O")

        if label_name.startswith("B-"):
            flush()
            ent_type = _label_to_entity_type(label_name)
            if ent_type is not None:
                current_type = ent_type
                current_word_indices = [i]
            continue

        if label_name.startswith("I-"):
            ent_type = _label_to_entity_type(label_name)
            if ent_type is None:
                flush()
                continue
            if current_type == ent_type and current_word_indices:
                current_word_indices.append(i)
            else:
                flush()
                current_type = ent_type
                current_word_indices = [i]
            continue

        flush()

    flush()
    return entities


def _distance_between_entities(ent_a: Dict[str, Any], ent_b: Dict[str, Any]) -> float:
    ax, ay = ent_a["center"]
    bx, by = ent_b["center"]
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def _choose_positive_answer(
    q_ent: Dict[str, Any],
    answer_entities: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not answer_entities:
        return None

    qx, qy = q_ent["center"]
    preferred = []
    for a in answer_entities:
        ax, ay = a["center"]
        if ax >= qx or ay >= qy:
            preferred.append(a)

    pool = preferred if preferred else list(answer_entities)
    best = min(pool, key=lambda a: _distance_between_entities(q_ent, a))
    return best


def _build_pseudo_related_pairs(
    word_label_ids: Sequence[int],
    id2label: Dict[int, str],
    boxes_1000: Sequence[Sequence[int]],
    negative_per_question: int = 2,
) -> List[Tuple[int, int, float]]:
    entities = _extract_entities_from_word_labels(word_label_ids, id2label, boxes_1000)
    questions = [e for e in entities if e["type"] == "question"]
    answers = [e for e in entities if e["type"] == "answer"]
    if not questions or not answers:
        return []

    pair_labels: Dict[Tuple[int, int], float] = {}

    for q in questions:
        pos = _choose_positive_answer(q, answers)
        if pos is None:
            continue

        q_idx = int(q["first_word"])
        pos_idx = int(pos["first_word"])
        pair_labels[(q_idx, pos_idx)] = 1.0

        neg_pool = [a for a in answers if int(a["first_word"]) != pos_idx]
        neg_pool.sort(key=lambda a: _distance_between_entities(q, a))
        for neg in neg_pool[: max(0, int(negative_per_question))]:
            n_idx = int(neg["first_word"])
            pair_labels.setdefault((q_idx, n_idx), 0.0)

    pairs = [(s, d, float(lbl)) for (s, d), lbl in sorted(pair_labels.items())]
    return pairs


def _positive_pair_set(pairs: Sequence[Tuple[int, int, float]]) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for s, d, y in pairs:
        if float(y) >= 0.5:
            out.add((int(s), int(d)))
    return out


def _heuristic_pred_related_pairs(
    pred_word_label_ids: Sequence[int],
    id2label: Dict[int, str],
    boxes_1000: Sequence[Sequence[int]],
) -> Set[Tuple[int, int]]:
    entities = _extract_entities_from_word_labels(pred_word_label_ids, id2label, boxes_1000)
    questions = [e for e in entities if e["type"] == "question"]
    answers = [e for e in entities if e["type"] == "answer"]
    out: Set[Tuple[int, int]] = set()
    if not questions or not answers:
        return out

    for q in questions:
        pos = _choose_positive_answer(q, answers)
        if pos is None:
            continue
        out.add((int(q["first_word"]), int(pos["first_word"])))
    return out


def _candidate_pairs_from_predictions(
    pred_word_label_ids: Sequence[int],
    id2label: Dict[int, str],
    boxes_1000: Sequence[Sequence[int]],
    max_answers_per_question: int = 8,
) -> List[Tuple[int, int, float]]:
    entities = _extract_entities_from_word_labels(pred_word_label_ids, id2label, boxes_1000)
    questions = [e for e in entities if e["type"] == "question"]
    answers = [e for e in entities if e["type"] == "answer"]

    if not questions or not answers:
        return []

    pairs: List[Tuple[int, int, float]] = []
    max_k = max(1, int(max_answers_per_question))
    for q in questions:
        ranked = sorted(answers, key=lambda a: _distance_between_entities(q, a))
        for a in ranked[:max_k]:
            pairs.append((int(q["first_word"]), int(a["first_word"]), 0.0))
    return pairs


# ---------------------------------------------------------------------------
# Processor / data preparation
# ---------------------------------------------------------------------------

def _load_processor(cache_dir: str, init_checkpoint: str, input_size: int) -> LayoutLMv3Processor:
    try:
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
            cache_dir=cache_dir,
        )
    except Exception:
        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(init_checkpoint, local_files_only=True)
        processor = LayoutLMv3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            apply_ocr=False,
        )

    processor.image_processor.size = {"height": input_size, "width": input_size}
    processor.image_processor.do_resize = True
    processor.image_processor.do_pad = True
    return processor


def _prepare_split(
    split,
    processor: LayoutLMv3Processor,
    backend: OCRBackend,
    id2label: Dict[int, str],
    o_label_id: int,
    input_size: int,
    limit: Optional[int] = None,
    use_gt_bboxes: bool = False,
) -> Tuple[List[PreparedSample], Dict[str, Any]]:
    samples: List[PreparedSample] = []
    truncation_hits = 0
    ocr_empty_docs = 0

    if limit is not None:
        split = split.select(range(min(int(limit), len(split))))

    for idx, ex in enumerate(split):
        print(f"[prepare] {idx + 1}/{len(split)}", end="\r")

        image = ex["image"].convert("RGB")
        gt_words = ex["words"]
        gt_boxes = ex["bboxes"]
        gt_labels = ex["ner_tags"]

        if use_gt_bboxes:
            words = gt_words
            boxes_1000 = gt_boxes
            confidences = [1.0] * len(words)
            aligned_labels = gt_labels
        else:
            words, boxes_1000, confidences, _ = backend.run(image)
            if not words:
                ocr_empty_docs += 1
                words = [""]
                boxes_1000 = [[0, 0, 0, 0]]
                confidences = [0.0]

            aligned_labels = align_ocr_to_ground_truth(
                words,
                boxes_1000,
                gt_words,
                gt_boxes,
                gt_labels,
                default_label_id=o_label_id,
            )

        image_sq = _pad_to_square(image, input_size)
        encoding = processor(
            image_sq,
            words,
            boxes=boxes_1000,
            word_labels=aligned_labels,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)

        max_wid = _max_word_id(word_ids)
        kept_words = max_wid + 1 if max_wid >= 0 else 0
        if kept_words <= 0:
            kept_words = min(1, len(words))
        if kept_words < len(words):
            truncation_hits += 1

        boxes_trim = [list(map(int, b)) for b in boxes_1000[:kept_words]]
        labels_trim = [int(x) for x in aligned_labels[:kept_words]]

        relation_pairs = _build_pseudo_related_pairs(
            labels_trim,
            id2label,
            boxes_trim,
            negative_per_question=2,
        )

        token_labels = encoding["labels"].squeeze(0).to(torch.long)
        inputs: Dict[str, torch.Tensor] = {}
        for k in ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"):
            if k in encoding:
                inputs[k] = encoding[k]

        conf_arr = np.array(confidences[:kept_words], dtype=np.float32) if confidences else np.array([0.0])
        low_conf_ratio = float((conf_arr < 0.5).mean()) if conf_arr.size > 0 else 1.0

        samples.append(
            PreparedSample(
                inputs=inputs,
                token_labels=token_labels,
                word_ids=list(word_ids),
                boxes_1000=boxes_trim,
                aligned_word_label_ids=labels_trim,
                relation_pairs=relation_pairs,
                low_conf_ratio=low_conf_ratio,
            )
        )

    print()

    stats = {
        "docs": len(samples),
        "truncation_hits": int(truncation_hits),
        "ocr_empty_docs": int(ocr_empty_docs),
    }
    return samples, stats


# ---------------------------------------------------------------------------
# Graph-enhanced model
# ---------------------------------------------------------------------------

class SinusoidalPositionalEmbedding2D(nn.Module):
    """
    Encodes 2D normalized coordinates [x1, y1, x2, y2] into a high-dimensional vector.
    """
    def __init__(self, d_model: int = 768, temperature: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        # Each coordinate (x1, y1, x2, y2) gets d_model // 4 dimensions
        self.d_c = d_model // 4

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: (N, 4) tensor with values in [0, 1]
        Returns:
            (N, d_model) positional embeddings
        """
        N = boxes.shape[0]
        device = boxes.device
        
        # dim_t = [0, 2, 4, ..., d_c - 2]
        dim_t = torch.arange(0, self.d_c, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.temperature ** (dim_t / self.d_c))
        
        # Reshape boxes to (N, 4, 1)
        boxes_exp = boxes.unsqueeze(-1)  # (N, 4, 1)
        # sin/cos (N, 4, d_c // 2)
        sin_inp = boxes_exp * inv_freq
        pos_sin = torch.sin(sin_inp)
        pos_cos = torch.cos(sin_inp)
        
        # Concatenate sin and cos -> (N, 4, d_c)
        pos_coord = torch.cat([pos_sin, pos_cos], dim=-1)
        
        # Flatten to (N, 4 * d_c) which is (N, d_model)
        pos_emb = pos_coord.view(N, -1)
        
        # If d_model % 4 != 0, pad with zeros
        if pos_emb.shape[1] < self.d_model:
            padding = torch.zeros((N, self.d_model - pos_emb.shape[1]), device=device)
            pos_emb = torch.cat([pos_emb, padding], dim=-1)
            
        return pos_emb


class GraphEnhancedLayoutLMv3(nn.Module):
    def __init__(
        self,
        init_checkpoint: str,
        num_labels: int,
        neighbor_k: int = 8,
        gcn_layers: int = 2,
        relation_embed_dim: int = 16,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        _require_pyg()

        self.init_checkpoint = init_checkpoint
        self.neighbor_k = int(neighbor_k)
        self.gcn_layers = int(gcn_layers)
        self.relation_embed_dim = int(relation_embed_dim)
        self.dropout_p = float(dropout)
        self.edge_feat_dim = 7

        self.backbone = LayoutLMv3ForTokenClassification.from_pretrained(
            init_checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            cache_dir=cache_dir,
        )
        hidden = int(self.backbone.config.hidden_size)

        self.pos_emb = SinusoidalPositionalEmbedding2D(d_model=hidden)
        self.relation_embedding = nn.Embedding(len(RELATION_NAME_TO_ID), self.relation_embed_dim)

        edge_dim = self.edge_feat_dim + self.relation_embed_dim
        self.graph_layers = nn.ModuleList(
            [
                GATv2Conv(
                    hidden,
                    hidden,
                    heads=1,
                    concat=False,
                    dropout=self.dropout_p,
                    edge_dim=edge_dim,
                )
                for _ in range(max(1, self.gcn_layers))
            ]
        )

        self.graph_dropout = nn.Dropout(self.dropout_p)

        relation_in = hidden * 2 + edge_dim
        self.relation_head = nn.Sequential(
            nn.Linear(relation_in, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(256, 1),
        )

        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.relation_loss_fn = nn.BCEWithLogitsLoss()

    @staticmethod
    def _aggregate_word_embeddings(
        token_hidden: torch.Tensor,
        word_ids: Sequence[Optional[int]],
        num_words: int,
    ) -> torch.Tensor:
        if num_words <= 0:
            return torch.zeros((0, token_hidden.shape[-1]), device=token_hidden.device, dtype=token_hidden.dtype)

        hidden = token_hidden.shape[-1]
        sums = torch.zeros((num_words, hidden), dtype=token_hidden.dtype, device=token_hidden.device)
        cnts = torch.zeros((num_words,), dtype=token_hidden.dtype, device=token_hidden.device)

        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid < 0 or wid >= num_words:
                continue
            sums[wid] += token_hidden[pos]
            cnts[wid] += 1.0

        cnts = cnts.clamp(min=1.0)
        return sums / cnts.unsqueeze(-1)

    @staticmethod
    def _fuse_tokens(
        token_hidden: torch.Tensor,
        graph_hidden: torch.Tensor,
        word_ids: Sequence[Optional[int]],
    ) -> torch.Tensor:
        if graph_hidden.numel() == 0:
            return token_hidden

        fused = token_hidden.clone()
        n_words = graph_hidden.shape[0]
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if 0 <= wid < n_words:
                fused[pos] = fused[pos] + graph_hidden[wid]
        return fused

    def score_relation_pairs(
        self,
        graph_hidden: torch.Tensor,
        boxes_1000: Sequence[Sequence[int]],
        relation_pairs: Optional[Sequence[Tuple[int, int, float]]] = None,
        ablate_relation_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, float]]]:
        if relation_pairs is None:
            relation_pairs = []

        clean_pairs: List[Tuple[int, int, float]] = []
        n_words = int(graph_hidden.shape[0])
        for item in relation_pairs:
            if len(item) >= 3:
                s, d, y = int(item[0]), int(item[1]), float(item[2])
            elif len(item) == 2:
                s, d = int(item[0]), int(item[1])
                y = 0.0
            else:
                continue
            if s < 0 or d < 0 or s >= n_words or d >= n_words:
                continue
            if s >= len(boxes_1000) or d >= len(boxes_1000):
                continue
            clean_pairs.append((s, d, y))

        if not clean_pairs:
            return (
                torch.zeros((0,), dtype=torch.float32, device=graph_hidden.device),
                torch.zeros((0,), dtype=torch.float32, device=graph_hidden.device),
                [],
            )

        src = torch.tensor([p[0] for p in clean_pairs], dtype=torch.long, device=graph_hidden.device)
        dst = torch.tensor([p[1] for p in clean_pairs], dtype=torch.long, device=graph_hidden.device)
        labels = torch.tensor([p[2] for p in clean_pairs], dtype=torch.float32, device=graph_hidden.device)

        rel_ids_list = []
        cont_list = []
        for s, d, _ in clean_pairs:
            rel_ids_list.append(_relation_id_from_boxes(boxes_1000[s], boxes_1000[d]))
            cont_list.append(_edge_cont_features(boxes_1000[s], boxes_1000[d]))

        rel_ids = torch.tensor(rel_ids_list, dtype=torch.long, device=graph_hidden.device)
        cont = torch.tensor(cont_list, dtype=torch.float32, device=graph_hidden.device)

        keep_mask = torch.ones_like(rel_ids, dtype=torch.bool)
        if ablate_relation_id is not None:
            keep_mask = rel_ids != int(ablate_relation_id)

        if keep_mask.sum().item() == 0:
            return (
                torch.zeros((0,), dtype=torch.float32, device=graph_hidden.device),
                torch.zeros((0,), dtype=torch.float32, device=graph_hidden.device),
                [],
            )

        src = src[keep_mask]
        dst = dst[keep_mask]
        labels = labels[keep_mask]
        rel_ids = rel_ids[keep_mask]
        cont = cont[keep_mask]

        kept_pairs = [p for p, keep in zip(clean_pairs, keep_mask.tolist()) if keep]

        rel_emb = self.relation_embedding(rel_ids)
        edge_feat = torch.cat([cont, rel_emb], dim=-1)

        src_h = graph_hidden[src]
        dst_h = graph_hidden[dst]
        pair_h = torch.cat([src_h, dst_h, edge_feat], dim=-1)
        logits = self.relation_head(pair_h).squeeze(-1)
        return logits, labels, kept_pairs

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        word_ids: Sequence[Optional[int]],
        boxes_1000: Sequence[Sequence[int]],
        relation_pairs: Optional[Sequence[Tuple[int, int, float]]] = None,
        ablate_relation_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "bbox": inputs["bbox"],
            "pixel_values": inputs["pixel_values"],
            "return_dict": True,
        }
        if "token_type_ids" in inputs:
            model_inputs["token_type_ids"] = inputs["token_type_ids"]

        outputs = self.backbone.layoutlmv3(**model_inputs)
        seq_len = int(inputs["input_ids"].shape[1])
        token_hidden = outputs[0].squeeze(0)[:seq_len]

        max_wid = _max_word_id(word_ids)
        num_words = max_wid + 1 if max_wid >= 0 else len(boxes_1000)
        num_words = min(num_words, len(boxes_1000))
        num_words = max(0, num_words)

        word_hidden = self._aggregate_word_embeddings(token_hidden, word_ids, num_words)

        if num_words > 0:
            # Normalized boxes [0, 1] for sinusoidal embedding
            boxes_norm = torch.tensor(boxes_1000[:num_words], dtype=torch.float32, device=token_hidden.device) / 1000.0
            pos_features = self.pos_emb(boxes_norm)
            word_hidden = word_hidden + pos_features

        graph_hidden = word_hidden
        edge_count = 0
        if num_words > 1:
            edge_index, edge_attr_cont, edge_rel_ids = _build_knn_graph(
                [list(map(int, b)) for b in boxes_1000[:num_words]],
                neighbor_k=self.neighbor_k,
                device=token_hidden.device,
            )

            if ablate_relation_id is not None and edge_rel_ids.numel() > 0:
                keep = edge_rel_ids != int(ablate_relation_id)
                edge_index = edge_index[:, keep]
                edge_attr_cont = edge_attr_cont[keep]
                edge_rel_ids = edge_rel_ids[keep]

            edge_count = int(edge_index.shape[1])
            if edge_count > 0:
                rel_emb = self.relation_embedding(edge_rel_ids)
                edge_attr = torch.cat([edge_attr_cont, rel_emb], dim=-1)

                h = word_hidden
                for conv in self.graph_layers:
                    h_new = conv(h, edge_index, edge_attr=edge_attr)
                    h_new = F.relu(h_new)
                    h_new = self.graph_dropout(h_new)
                    h = h + h_new
                graph_hidden = h

        fused_tokens = self._fuse_tokens(token_hidden, graph_hidden, word_ids)
        seq = self.backbone.dropout(fused_tokens.unsqueeze(0))
        logits = self.backbone.classifier(seq)

        rel_logits, rel_labels, used_pairs = self.score_relation_pairs(
            graph_hidden=graph_hidden,
            boxes_1000=boxes_1000[:num_words],
            relation_pairs=relation_pairs,
            ablate_relation_id=ablate_relation_id,
        )

        return {
            "logits": logits,
            "graph_hidden": graph_hidden,
            "relation_logits": rel_logits,
            "relation_labels": rel_labels,
            "relation_pairs": used_pairs,
            "num_words": num_words,
            "edge_count": edge_count,
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_num(path: str) -> int:
    base = os.path.basename(path)
    if not base.startswith("checkpoint-"):
        return -1
    try:
        return int(base.rsplit("-", 1)[-1])
    except Exception:
        return -1


def _find_best_graph_checkpoint(model_dir: str) -> Optional[str]:
    if not os.path.isdir(model_dir):
        return None

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
            metric = float(state.get("best_metric", float("-inf")))
            if metric > best_metric:
                best_metric = metric
                best_ckpt = os.path.join(model_dir, sub)
        except Exception:
            continue

    if best_ckpt is not None:
        return best_ckpt

    ckpts = [
        os.path.join(model_dir, d)
        for d in os.listdir(model_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d))
    ]
    if not ckpts:
        return None
    return max(ckpts, key=_checkpoint_num)


def _save_graph_checkpoint(
    model: GraphEnhancedLayoutLMv3,
    checkpoint_dir: str,
    metric_f1: float,
    epoch: int,
    metadata: Dict[str, Any],
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, os.path.join(checkpoint_dir, "graph_model.pt"))

    trainer_state = {
        "best_metric": float(metric_f1),
        "epoch": int(epoch),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)

    with open(os.path.join(checkpoint_dir, "graph_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_graph_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
    cache_dir: str,
) -> Tuple[GraphEnhancedLayoutLMv3, Dict[str, Any]]:
    ckpt_path = os.path.join(checkpoint_dir, "graph_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Graph checkpoint file missing: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device)
    metadata = payload.get("metadata", {})

    init_checkpoint = metadata.get("init_checkpoint", DEFAULT_CHECKPOINT)
    num_labels = int(metadata.get("num_labels", 7))
    neighbor_k = int(metadata.get("neighbor_k", 8))
    gcn_layers = int(metadata.get("gcn_layers", 2))
    relation_embed_dim = int(metadata.get("relation_embed_dim", 16))
    dropout = float(metadata.get("dropout", 0.1))

    model = GraphEnhancedLayoutLMv3(
        init_checkpoint=init_checkpoint,
        num_labels=num_labels,
        neighbor_k=neighbor_k,
        gcn_layers=gcn_layers,
        relation_embed_dim=relation_embed_dim,
        dropout=dropout,
        cache_dir=cache_dir,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, metadata


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _collapse_word_predictions(
    pred_ids: Sequence[int],
    token_labels: Sequence[int],
    word_ids: Sequence[Optional[int]],
    label_list: Sequence[str],
    o_label_id: int,
) -> Tuple[List[str], List[str], List[int]]:
    max_wid = _max_word_id(word_ids)
    num_words = max_wid + 1 if max_wid >= 0 else 0
    word_pred_ids = [int(o_label_id)] * max(0, num_words)

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


def _compute_entity_metrics(
    seqeval_metric,
    true_all: List[List[str]],
    pred_all: List[List[str]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    result = seqeval_metric.compute(predictions=pred_all, references=true_all)
    overall = {
        "precision": round(float(result["overall_precision"]), 4),
        "recall": round(float(result["overall_recall"]), 4),
        "f1": round(float(result["overall_f1"]), 4),
        "accuracy": round(float(result["overall_accuracy"]), 4),
    }

    per_class: Dict[str, Dict[str, float]] = {}
    for key, val in result.items():
        if not isinstance(val, dict):
            continue
        per_class[key] = {
            "precision": round(float(val.get("precision", 0.0)), 4),
            "recall": round(float(val.get("recall", 0.0)), 4),
            "f1": round(float(val.get("f1", 0.0)), 4),
            "number": int(val.get("number", 0)),
        }

    return overall, per_class


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _to_device_inputs(sample: PreparedSample, device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in sample.inputs.items()}


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def _evaluate_baseline_model(
    model: LayoutLMv3ForTokenClassification,
    samples: Sequence[PreparedSample],
    label_list: Sequence[str],
    id2label: Dict[int, str],
    o_label_id: int,
    metric_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    seqeval_metric = hf_evaluate.load("seqeval", cache_dir=metric_cache_dir)
    true_all: List[List[str]] = []
    pred_all: List[List[str]] = []

    rel_tp = rel_fp = rel_fn = 0
    latencies_ms: List[float] = []

    for sample in samples:
        inputs = _to_device_inputs(sample, next(model.parameters()).device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(0)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        pred_ids = logits.argmax(-1).cpu().tolist()
        token_labels = sample.token_labels.cpu().tolist()

        doc_true, doc_pred, word_pred_ids = _collapse_word_predictions(
            pred_ids=pred_ids,
            token_labels=token_labels,
            word_ids=sample.word_ids,
            label_list=label_list,
            o_label_id=o_label_id,
        )
        true_all.append(doc_true)
        pred_all.append(doc_pred)

        gt_pairs = _positive_pair_set(sample.relation_pairs)
        pred_pairs = _heuristic_pred_related_pairs(word_pred_ids, id2label, sample.boxes_1000)

        rel_tp += len(pred_pairs & gt_pairs)
        rel_fp += len(pred_pairs - gt_pairs)
        rel_fn += len(gt_pairs - pred_pairs)

    overall, per_class = _compute_entity_metrics(seqeval_metric, true_all, pred_all)
    return {
        "entity": overall,
        "per_class": per_class,
        "related_field": _prf(rel_tp, rel_fp, rel_fn),
        "avg_model_latency_ms": round(float(statistics.mean(latencies_ms)) if latencies_ms else 0.0, 3),
        "docs": len(samples),
    }


def _evaluate_graph_model(
    model: GraphEnhancedLayoutLMv3,
    samples: Sequence[PreparedSample],
    label_list: Sequence[str],
    id2label: Dict[int, str],
    o_label_id: int,
    edge_threshold: float,
    ablate_relation_id: Optional[int] = None,
    metric_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    seqeval_metric = hf_evaluate.load("seqeval", cache_dir=metric_cache_dir)
    true_all: List[List[str]] = []
    pred_all: List[List[str]] = []

    rel_tp = rel_fp = rel_fn = 0
    latencies_ms: List[float] = []
    edge_counts: List[int] = []

    for sample in samples:
        inputs = _to_device_inputs(sample, next(model.parameters()).device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                inputs=inputs,
                word_ids=sample.word_ids,
                boxes_1000=sample.boxes_1000,
                relation_pairs=None,
                ablate_relation_id=ablate_relation_id,
            )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        edge_counts.append(int(out.get("edge_count", 0)))

        pred_ids = out["logits"].argmax(-1).squeeze(0).cpu().tolist()
        token_labels = sample.token_labels.cpu().tolist()

        doc_true, doc_pred, word_pred_ids = _collapse_word_predictions(
            pred_ids=pred_ids,
            token_labels=token_labels,
            word_ids=sample.word_ids,
            label_list=label_list,
            o_label_id=o_label_id,
        )
        true_all.append(doc_true)
        pred_all.append(doc_pred)

        candidates = _candidate_pairs_from_predictions(
            pred_word_label_ids=word_pred_ids,
            id2label=id2label,
            boxes_1000=sample.boxes_1000,
            max_answers_per_question=8,
        )

        pred_pairs: Set[Tuple[int, int]] = set()
        if candidates:
            with torch.no_grad():
                rel_logits, _, kept_pairs = model.score_relation_pairs(
                    graph_hidden=out["graph_hidden"],
                    boxes_1000=sample.boxes_1000,
                    relation_pairs=candidates,
                    ablate_relation_id=ablate_relation_id,
                )
            if rel_logits.numel() > 0:
                probs = torch.sigmoid(rel_logits).cpu().tolist()
                for pair, p in zip(kept_pairs, probs):
                    if float(p) >= float(edge_threshold):
                        pred_pairs.add((int(pair[0]), int(pair[1])))

        gt_pairs = _positive_pair_set(sample.relation_pairs)
        rel_tp += len(pred_pairs & gt_pairs)
        rel_fp += len(pred_pairs - gt_pairs)
        rel_fn += len(gt_pairs - pred_pairs)

    overall, per_class = _compute_entity_metrics(seqeval_metric, true_all, pred_all)
    return {
        "entity": overall,
        "per_class": per_class,
        "related_field": _prf(rel_tp, rel_fp, rel_fn),
        "avg_model_latency_ms": round(float(statistics.mean(latencies_ms)) if latencies_ms else 0.0, 3),
        "avg_graph_edges": round(float(statistics.mean(edge_counts)) if edge_counts else 0.0, 2),
        "docs": len(samples),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_graph_model(
    model: GraphEnhancedLayoutLMv3,
    train_samples: Sequence[PreparedSample],
    eval_samples: Sequence[PreparedSample],
    label_list: Sequence[str],
    id2label: Dict[int, str],
    o_label_id: int,
    args,
    output_model_dir: str,
    metric_cache_dir: Optional[str],
) -> Dict[str, Any]:
    if int(args.batch_size) != 1:
        print("[train] batch_size>1 is not supported in this pipeline; forcing batch_size=1.")

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
    best_f1 = float("-inf")
    best_ckpt = None
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = list(range(len(train_samples)))
        random.shuffle(order)

        loss_total = 0.0
        loss_token_total = 0.0
        loss_rel_total = 0.0

        for step, idx in enumerate(order, start=1):
            sample = train_samples[idx]
            inputs = _to_device_inputs(sample, device)
            token_labels = sample.token_labels.to(device)

            out = model(
                inputs=inputs,
                word_ids=sample.word_ids,
                boxes_1000=sample.boxes_1000,
                relation_pairs=sample.relation_pairs,
                ablate_relation_id=None,
            )

            token_loss = model.token_loss_fn(
                out["logits"].view(-1, out["logits"].shape[-1]),
                token_labels.view(-1),
            )

            if out["relation_logits"].numel() > 0:
                relation_loss = model.relation_loss_fn(out["relation_logits"], out["relation_labels"])
            else:
                relation_loss = torch.zeros((), dtype=torch.float32, device=device)

            loss = token_loss + float(args.relation_loss_weight) * relation_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_total += float(loss.item())
            loss_token_total += float(token_loss.item())
            loss_rel_total += float(relation_loss.item())

            if step % 25 == 0 or step == len(order):
                print(
                    f"[train] epoch={epoch} step={step}/{len(order)} "
                    f"loss={loss.item():.4f} token={token_loss.item():.4f} rel={relation_loss.item():.4f}"
                )

        model.eval()
        eval_report = _evaluate_graph_model(
            model=model,
            samples=eval_samples,
            label_list=label_list,
            id2label=id2label,
            o_label_id=o_label_id,
            edge_threshold=float(args.edge_threshold),
            ablate_relation_id=None,
            metric_cache_dir=metric_cache_dir,
        )

        eval_f1 = float(eval_report["entity"]["f1"])
        history_item = {
            "epoch": int(epoch),
            "train_loss": round(loss_total / max(1, len(order)), 6),
            "train_token_loss": round(loss_token_total / max(1, len(order)), 6),
            "train_relation_loss": round(loss_rel_total / max(1, len(order)), 6),
            "eval_entity_f1": round(eval_f1, 4),
            "eval_related_field_f1": round(float(eval_report["related_field"]["f1"]), 4),
        }
        history.append(history_item)

        print(
            f"[eval] epoch={epoch} entity_f1={eval_report['entity']['f1']:.4f} "
            f"related_f1={eval_report['related_field']['f1']:.4f}"
        )

        checkpoint_dir = os.path.join(output_model_dir, f"checkpoint-{epoch}")
        metadata = {
            "init_checkpoint": model.init_checkpoint,
            "num_labels": int(model.backbone.config.num_labels),
            "neighbor_k": int(model.neighbor_k),
            "gcn_layers": int(model.gcn_layers),
            "relation_embed_dim": int(model.relation_embed_dim),
            "dropout": float(model.dropout_p),
            "input_size": int(getattr(model.backbone.config, "input_size", 384)),
            "label_list": list(label_list),
            "epoch": int(epoch),
            "eval_entity_f1": float(eval_f1),
            "eval_related_field_f1": float(eval_report["related_field"]["f1"]),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        }
        _save_graph_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            metric_f1=eval_f1,
            epoch=epoch,
            metadata=metadata,
        )

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_ckpt = checkpoint_dir

    return {
        "best_checkpoint": best_ckpt,
        "best_entity_f1": round(float(best_f1), 4) if best_ckpt else None,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Public evaluator for eval_layoutlm_all.py integration
# ---------------------------------------------------------------------------

def evaluate_layoutgraph_pair(
    model_dir: str,
    ocr_engine: str,
    dataset,
    cache_dir: str,
    limit: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if ocr_engine != "doctr":
        raise ValueError("layoutgraph variation currently supports only docTR OCR.")

    _require_pyg()

    if device is None:
        device = _pick_device("auto")

    abs_model_dir = os.path.join(SCRIPT_DIR, model_dir)
    checkpoint = _find_best_graph_checkpoint(abs_model_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No graph checkpoint found under {abs_model_dir}. "
            "Train with `python LayoutLM/layout_graph_pipeline.py train ...` first."
        )

    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    o_label_id = int({label: i for i, label in enumerate(label_list)}.get("O", 0))

    # Input size from checkpoint metadata when available.
    metadata_path = os.path.join(checkpoint, "graph_config.json")
    input_size = 384
    init_checkpoint = DEFAULT_CHECKPOINT
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        input_size = int(metadata.get("input_size", 384))
        init_checkpoint = metadata.get("init_checkpoint", DEFAULT_CHECKPOINT)

    processor = _load_processor(cache_dir=cache_dir, init_checkpoint=init_checkpoint, input_size=input_size)
    backend = OCRBackend(engine="doctr")

    print("  Preparing docTR samples for layoutgraph evaluation ...")
    samples, prep_stats = _prepare_split(
        split=dataset["test"],
        processor=processor,
        backend=backend,
        id2label=id2label,
        o_label_id=o_label_id,
        input_size=input_size,
        limit=limit,
    )

    print(f"  Prepared docs: {prep_stats['docs']} (truncated={prep_stats['truncation_hits']})")

    model, _ = _load_graph_checkpoint(checkpoint, device=device, cache_dir=cache_dir)
    t0 = time.time()
    report = _evaluate_graph_model(
        model=model,
        samples=samples,
        label_list=label_list,
        id2label=id2label,
        o_label_id=o_label_id,
        edge_threshold=0.5,
        ablate_relation_id=None,
        metric_cache_dir=cache_dir,
    )
    elapsed = time.time() - t0

    return {
        "model_dir": model_dir,
        "ocr_engine": ocr_engine,
        "checkpoint": os.path.relpath(checkpoint, SCRIPT_DIR),
        "overall": report["entity"],
        "per_class": report["per_class"],
        "total_docs": int(report["docs"]),
        "total_gt_words": 0,
        "total_ocr_words": 0,
        "elapsed_seconds": round(float(elapsed), 2),
        "seconds_per_doc": round(float(elapsed / max(1, report["docs"])), 3),
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _prepare_common(
    args,
    include_train: bool,
) -> Dict[str, Any]:
    _require_pyg()

    _set_seed(int(args.seed))
    cache_dir = _resolve_cache_dir(args.cache_dir)
    init_checkpoint = _resolve_path(args.init_checkpoint, DEFAULT_CHECKPOINT)
    output_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    output_model_dir = _resolve_path(args.model_output_dir, DEFAULT_MODEL_DIR)
    device = _pick_device(args.device)

    if args.ocr_engine != "doctr":
        raise ValueError("This pipeline is docTR-only. Use --ocr_engine doctr.")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    print("[setup] Loading FUNSD dataset ...")
    dataset = load_dataset("nielsr/funsd", cache_dir=cache_dir)

    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    o_label_id = int(label2id.get("O", 0))

    input_size = _read_input_size(init_checkpoint)
    processor = _load_processor(cache_dir=cache_dir, init_checkpoint=init_checkpoint, input_size=input_size)
    backend = OCRBackend(engine="doctr")

    print("[setup] Preparing FUNSD test split with docTR OCR ...")
    test_samples, test_stats = _prepare_split(
        split=dataset["test"],
        processor=processor,
        backend=backend,
        id2label=id2label,
        o_label_id=o_label_id,
        input_size=input_size,
        limit=args.limit,
        use_gt_bboxes=bool(getattr(args, "use_gt_bboxes", False)),
    )

    train_samples = None
    train_stats = None
    if include_train:
        print("[setup] Preparing FUNSD train split with docTR OCR ...")
        train_samples, train_stats = _prepare_split(
            split=dataset["train"],
            processor=processor,
            backend=backend,
            id2label=id2label,
            o_label_id=o_label_id,
            input_size=input_size,
            limit=args.limit,
            use_gt_bboxes=bool(getattr(args, "use_gt_bboxes", False)),
        )

    return {
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "output_model_dir": output_model_dir,
        "device": device,
        "init_checkpoint": init_checkpoint,
        "input_size": input_size,
        "label_list": label_list,
        "id2label": id2label,
        "o_label_id": o_label_id,
        "test_samples": test_samples,
        "test_stats": test_stats,
        "train_samples": train_samples,
        "train_stats": train_stats,
    }


def _handle_train(args) -> Dict[str, Any]:
    common = _prepare_common(args, include_train=True)

    model = GraphEnhancedLayoutLMv3(
        init_checkpoint=common["init_checkpoint"],
        num_labels=len(common["label_list"]),
        neighbor_k=int(args.neighbor_k),
        gcn_layers=int(args.gcn_layers),
        relation_embed_dim=16,
        dropout=0.1,
        cache_dir=common["cache_dir"],
    ).to(common["device"])

    print("[train] Starting graph-enhanced training ...")
    train_report = _train_graph_model(
        model=model,
        train_samples=common["train_samples"],
        eval_samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        args=args,
        output_model_dir=common["output_model_dir"],
        metric_cache_dir=common["cache_dir"],
    )

    payload = {
        "status": "ok",
        "mode": "train",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "output_model_dir": common["output_model_dir"],
        "train_samples": len(common["train_samples"] or []),
        "eval_samples": len(common["test_samples"] or []),
        "train_stats": common["train_stats"],
        "eval_stats": common["test_stats"],
        "train_report": train_report,
    }
    return payload


def _load_or_train_for_eval(args, common: Dict[str, Any]) -> Tuple[GraphEnhancedLayoutLMv3, str]:
    checkpoint = args.checkpoint
    if checkpoint:
        ckpt = _resolve_path(checkpoint)
        model, _ = _load_graph_checkpoint(ckpt, device=common["device"], cache_dir=common["cache_dir"])
        return model, ckpt

    best = _find_best_graph_checkpoint(common["output_model_dir"])
    if best is None:
        raise FileNotFoundError(
            f"No checkpoint found in {common['output_model_dir']}. "
            "Run `train` first or pass --checkpoint."
        )
    model, _ = _load_graph_checkpoint(best, device=common["device"], cache_dir=common["cache_dir"])
    return model, best


def _handle_evaluate(args) -> Dict[str, Any]:
    common = _prepare_common(args, include_train=False)

    graph_model, graph_ckpt = _load_or_train_for_eval(args, common)
    baseline = LayoutLMv3ForTokenClassification.from_pretrained(common["init_checkpoint"]).to(common["device"])
    baseline.eval()

    print("[eval] Evaluating baseline model ...")
    baseline_report = _evaluate_baseline_model(
        model=baseline,
        samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        metric_cache_dir=common["cache_dir"],
    )

    print("[eval] Evaluating graph-enhanced model ...")
    graph_report = _evaluate_graph_model(
        model=graph_model,
        samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        edge_threshold=float(args.edge_threshold),
        ablate_relation_id=None,
        metric_cache_dir=common["cache_dir"],
    )

    payload = {
        "status": "ok",
        "mode": "evaluate",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": graph_ckpt,
        "baseline": baseline_report,
        "graph": graph_report,
        "delta_entity_f1": round(float(graph_report["entity"]["f1"] - baseline_report["entity"]["f1"]), 4),
        "delta_related_field_f1": round(
            float(graph_report["related_field"]["f1"] - baseline_report["related_field"]["f1"]),
            4,
        ),
    }
    return payload


def _handle_relation_eval(args) -> Dict[str, Any]:
    full = _handle_evaluate(args)
    return {
        "status": full["status"],
        "mode": "relation_eval",
        "timestamp_utc": full["timestamp_utc"],
        "checkpoint": full["checkpoint"],
        "baseline_related_field": full["baseline"]["related_field"],
        "graph_related_field": full["graph"]["related_field"],
        "delta_related_field_f1": full["delta_related_field_f1"],
    }


def _run_importance(
    graph_model: GraphEnhancedLayoutLMv3,
    samples: Sequence[PreparedSample],
    label_list: Sequence[str],
    id2label: Dict[int, str],
    o_label_id: int,
    edge_threshold: float,
    metric_cache_dir: Optional[str],
) -> Dict[str, Any]:
    base = _evaluate_graph_model(
        model=graph_model,
        samples=samples,
        label_list=label_list,
        id2label=id2label,
        o_label_id=o_label_id,
        edge_threshold=edge_threshold,
        ablate_relation_id=None,
        metric_cache_dir=metric_cache_dir,
    )

    ablations = []
    for rel_name, rel_id in RELATION_NAME_TO_ID.items():
        rep = _evaluate_graph_model(
            model=graph_model,
            samples=samples,
            label_list=label_list,
            id2label=id2label,
            o_label_id=o_label_id,
            edge_threshold=edge_threshold,
            ablate_relation_id=rel_id,
            metric_cache_dir=metric_cache_dir,
        )
        ablations.append(
            {
                "relation": rel_name,
                "entity_f1": rep["entity"]["f1"],
                "related_field_f1": rep["related_field"]["f1"],
                "delta_entity_f1_vs_full": round(float(rep["entity"]["f1"] - base["entity"]["f1"]), 4),
                "delta_related_field_f1_vs_full": round(
                    float(rep["related_field"]["f1"] - base["related_field"]["f1"]),
                    4,
                ),
            }
        )

    return {
        "full_graph": {
            "entity_f1": base["entity"]["f1"],
            "related_field_f1": base["related_field"]["f1"],
        },
        "ablations": ablations,
    }


def _handle_importance(args) -> Dict[str, Any]:
    common = _prepare_common(args, include_train=False)
    graph_model, graph_ckpt = _load_or_train_for_eval(args, common)

    importance_report = _run_importance(
        graph_model=graph_model,
        samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        edge_threshold=float(args.edge_threshold),
        metric_cache_dir=common["cache_dir"],
    )

    payload = {
        "status": "ok",
        "mode": "importance",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": graph_ckpt,
        "importance": importance_report,
    }
    return payload


def _handle_run_all(args) -> Dict[str, Any]:
    common = _prepare_common(args, include_train=True)

    best_ckpt = _find_best_graph_checkpoint(common["output_model_dir"])
    train_report = None
    if best_ckpt is None:
        print("[run_all] No existing graph checkpoint found. Training ...")
        model = GraphEnhancedLayoutLMv3(
            init_checkpoint=common["init_checkpoint"],
            num_labels=len(common["label_list"]),
            neighbor_k=int(args.neighbor_k),
            gcn_layers=int(args.gcn_layers),
            relation_embed_dim=16,
            dropout=0.1,
            cache_dir=common["cache_dir"],
        ).to(common["device"])
        train_report = _train_graph_model(
            model=model,
            train_samples=common["train_samples"],
            eval_samples=common["test_samples"],
            label_list=common["label_list"],
            id2label=common["id2label"],
            o_label_id=common["o_label_id"],
            args=args,
            output_model_dir=common["output_model_dir"],
            metric_cache_dir=common["cache_dir"],
        )
        best_ckpt = train_report["best_checkpoint"]
    else:
        print(f"[run_all] Reusing checkpoint: {best_ckpt}")

    if best_ckpt is None:
        raise RuntimeError("Unable to locate or train a graph checkpoint.")

    graph_model, _ = _load_graph_checkpoint(best_ckpt, device=common["device"], cache_dir=common["cache_dir"])
    baseline = LayoutLMv3ForTokenClassification.from_pretrained(common["init_checkpoint"]).to(common["device"])
    baseline.eval()

    print("[run_all] Evaluating baseline ...")
    baseline_report = _evaluate_baseline_model(
        model=baseline,
        samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        metric_cache_dir=common["cache_dir"],
    )

    print("[run_all] Evaluating graph model ...")
    graph_report = _evaluate_graph_model(
        model=graph_model,
        samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        edge_threshold=float(args.edge_threshold),
        ablate_relation_id=None,
        metric_cache_dir=common["cache_dir"],
    )

    low_conf_samples = [s for s in common["test_samples"] if float(s.low_conf_ratio) >= 0.2]
    low_conf_report = {
        "criteria": "ocr_low_conf_ratio>=0.2 (confidence<0.5)",
        "docs": len(low_conf_samples),
    }
    if low_conf_samples:
        print("[run_all] Evaluating low-confidence OCR subset ...")
        low_conf_report["baseline"] = _evaluate_baseline_model(
            model=baseline,
            samples=low_conf_samples,
            label_list=common["label_list"],
            id2label=common["id2label"],
            o_label_id=common["o_label_id"],
            metric_cache_dir=common["cache_dir"],
        )
        low_conf_report["graph"] = _evaluate_graph_model(
            model=graph_model,
            samples=low_conf_samples,
            label_list=common["label_list"],
            id2label=common["id2label"],
            o_label_id=common["o_label_id"],
            edge_threshold=float(args.edge_threshold),
            ablate_relation_id=None,
            metric_cache_dir=common["cache_dir"],
        )

    print("[run_all] Running relation-type importance ablations ...")
    importance_report = _run_importance(
        graph_model=graph_model,
        samples=common["test_samples"],
        label_list=common["label_list"],
        id2label=common["id2label"],
        o_label_id=common["o_label_id"],
        edge_threshold=float(args.edge_threshold),
        metric_cache_dir=common["cache_dir"],
    )

    payload = {
        "status": "ok",
        "mode": "run_all",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": best_ckpt,
        "init_checkpoint": common["init_checkpoint"],
        "device": str(common["device"]),
        "input_size": common["input_size"],
        "train_report": train_report,
        "test_stats": common["test_stats"],
        "baseline": baseline_report,
        "graph": graph_report,
        "delta_entity_f1": round(float(graph_report["entity"]["f1"] - baseline_report["entity"]["f1"]), 4),
        "delta_related_field_f1": round(
            float(graph_report["related_field"]["f1"] - baseline_report["related_field"]["f1"]),
            4,
        ),
        "ocr_corruption_analysis": low_conf_report,
        "relation_importance": importance_report,
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone layout-graph variation pipeline for LayoutLMv3 (docTR + FUNSD).")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="HF cache directory")
        p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Report/artifacts output directory")
        p.add_argument("--model_output_dir", type=str, default=DEFAULT_MODEL_DIR, help="Directory for graph model checkpoints")
        p.add_argument("--init_checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Base LayoutLMv3 checkpoint")
        p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Execution device")
        p.add_argument("--limit", type=int, default=None, help="Optional document cap for smoke tests")
        p.add_argument("--ocr_engine", type=str, default="doctr", choices=["doctr"], help="OCR engine (fixed to docTR)")
        p.add_argument("--use_gt_bboxes", type=bool, default=False, help="Use ground-truth words/boxes instead of OCR")
        p.add_argument("--seed", type=int, default=42, help="Random seed")

    def add_train_args(p):
        p.add_argument("--epochs", type=int, default=12, help="Training epochs")
        p.add_argument("--batch_size", type=int, default=1, help="Batch size (fixed to 1 in this pipeline)")
        p.add_argument("--lr", type=float, default=1.5e-5, help="Learning rate")
        p.add_argument("--gcn_layers", type=int, default=2, help="Number of graph convolution layers")
        p.add_argument("--neighbor_k", type=int, default=8, help="Directed top-k spatial neighbors")
        p.add_argument("--relation_loss_weight", type=float, default=0.3, help="Weight for relation BCE loss")
        p.add_argument("--edge_threshold", type=float, default=0.5, help="Q->A relation threshold")

    p_train = sub.add_parser("train", help="Train graph-enhanced LayoutLMv3 variation")
    add_common(p_train)
    add_train_args(p_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate baseline vs graph-enhanced model")
    add_common(p_eval)
    p_eval.add_argument("--checkpoint", type=str, default=None, help="Specific graph checkpoint directory")
    p_eval.add_argument("--edge_threshold", type=float, default=0.5, help="Q->A relation threshold")

    p_rel = sub.add_parser("relation_eval", help="Evaluate related-field (Q->A) metrics")
    add_common(p_rel)
    p_rel.add_argument("--checkpoint", type=str, default=None, help="Specific graph checkpoint directory")
    p_rel.add_argument("--edge_threshold", type=float, default=0.5, help="Q->A relation threshold")

    p_imp = sub.add_parser("importance", help="Run relation-type ablation importance analysis")
    add_common(p_imp)
    p_imp.add_argument("--checkpoint", type=str, default=None, help="Specific graph checkpoint directory")
    p_imp.add_argument("--edge_threshold", type=float, default=0.5, help="Q->A relation threshold")

    p_all = sub.add_parser("run_all", help="Train-if-needed + full eval + importance + corruption analysis")
    add_common(p_all)
    add_train_args(p_all)

    return parser


def _print_run_summary(payload: Dict[str, Any]) -> None:
    mode = payload.get("mode", "unknown")
    print(f"\n=== {mode.upper()} SUMMARY ===")

    if mode in {"evaluate", "run_all"}:
        b = payload["baseline"]
        g = payload["graph"]
        print("model      | entity_f1 | related_f1")
        print("------------------------------------")
        print(f"baseline   | {b['entity']['f1']:.4f}   | {b['related_field']['f1']:.4f}")
        print(f"layoutgraph| {g['entity']['f1']:.4f}   | {g['related_field']['f1']:.4f}")
        print(f"delta_f1   | {payload['delta_entity_f1']:+.4f}   | {payload['delta_related_field_f1']:+.4f}")

    if mode == "relation_eval":
        b = payload["baseline_related_field"]
        g = payload["graph_related_field"]
        print("related-field Q->A")
        print(f"baseline_f1   : {b['f1']:.4f}")
        print(f"layoutgraph_f1: {g['f1']:.4f}")
        print(f"delta_f1      : {payload['delta_related_field_f1']:+.4f}")

    if mode == "train":
        tr = payload.get("train_report", {})
        print(f"best_checkpoint: {tr.get('best_checkpoint')}")
        print(f"best_entity_f1 : {tr.get('best_entity_f1')}")

    if mode == "importance":
        imp = payload.get("importance", {})
        full = imp.get("full_graph", {})
        print(f"full_graph_entity_f1 : {full.get('entity_f1')}")
        print(f"full_graph_related_f1: {full.get('related_field_f1')}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            payload = _handle_train(args)
        elif args.command == "evaluate":
            payload = _handle_evaluate(args)
        elif args.command == "relation_eval":
            payload = _handle_relation_eval(args)
        elif args.command == "importance":
            payload = _handle_importance(args)
        elif args.command == "run_all":
            payload = _handle_run_all(args)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as e:
        payload = {
            "status": "error",
            "mode": args.command,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "error": str(e),
        }

    out_dir = _resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "layout_graph_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps({"status": payload.get("status", "unknown"), "report_path": out_path}, indent=2))
    _print_run_summary(payload)


if __name__ == "__main__":
    main()
