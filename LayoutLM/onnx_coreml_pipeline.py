"""
onnx_coreml_pipeline.py
=======================
Isolated ONNX/CoreML experimentation pipeline for LayoutLMv3.

Subcommands:
  - export_onnx
  - optimize_sparse_spatial
  - quantize_int8
  - export_coreml
  - benchmark
  - sim_test
  - run_all
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import onnx
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "layoutlmv3-funsd-doctr", "checkpoint-800")
DEFAULT_CACHE_DIR = os.getenv("HF_CACHE_DIR") or os.path.join(REPO_ROOT, ".hf_cache")
DEFAULT_ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "onnx_artifacts")
DEFAULT_OPSET = 17
DEFAULT_BUCKET_BOUNDS = [64, 128, 256, 384, 512]
DEFAULT_QUANT_OP_TYPES = ["MatMul", "Gemm"]


def _optional_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["_"])
    except Exception:
        return None


def _pick_onnx_providers(ort_mod, onnx_provider: str) -> List[str]:
    available = list(getattr(ort_mod, "get_available_providers", lambda: [])())
    if onnx_provider == "cpu":
        return ["CPUExecutionProvider"]
    if onnx_provider == "coreml":
        if "CoreMLExecutionProvider" not in available:
            return ["CPUExecutionProvider"]
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if onnx_provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    providers: List[str] = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _free_memory() -> None:
    """Release Python-held memory and flush GPU/MPS caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _select_logits_tensor(candidates: List[Any], num_labels: int) -> np.ndarray:
    # Deduplicate: convert to numpy first, skip identical objects
    seen_ids: set = set()
    arrays: List[np.ndarray] = []
    for x in candidates:
        if x is None or id(x) in seen_ids:
            continue
        seen_ids.add(id(x))
        arrays.append(np.asarray(x))
    for arr in arrays:
        if arr.ndim == 3 and arr.shape[-1] == num_labels:
            return arr
    for arr in arrays:
        if arr.ndim == 3 and arr.shape[1] == num_labels:
            return np.transpose(arr, (0, 2, 1))
    for arr in arrays:
        if arr.ndim == 2 and arr.shape[-1] == num_labels:
            return arr[None, ...]
    for arr in arrays:
        if arr.ndim == 2 and arr.shape[0] == num_labels:
            return np.transpose(arr, (1, 0))[None, ...]
    for arr in arrays:
        if arr.ndim == 3:
            return arr
    if not arrays:
        raise RuntimeError("No tensor outputs were produced by backend.")
    return arrays[0]


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
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _resolve_path(path: str, default: Optional[str] = None) -> str:
    candidate = path or default
    if candidate is None:
        raise ValueError("No path provided.")
    candidate = os.path.expanduser(candidate)
    if os.path.isabs(candidate):
        return candidate
    return os.path.abspath(os.path.join(REPO_ROOT, candidate))


def _read_input_size(checkpoint_dir: str) -> int:
    cfg = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(cfg):
        return 384
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        return int(data.get("input_size", 384))
    except Exception:
        return 384


def _pad_to_square(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((target - nw) // 2, (target - nh) // 2))
    return canvas


class SparseSpatialLayoutLMv3Wrapper(torch.nn.Module):
    """
    Wrapper that preserves model outputs while sparsifying bbox signal
    to active tokens (attention_mask == 1). This wrapper is shared by
    both ONNX export and CoreML conversion.
    """

    def __init__(self, model: LayoutLMv3ForTokenClassification):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=bbox.dtype)
        sparse_bbox = bbox * mask
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=sparse_bbox,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


def _load_model_and_processor(
    checkpoint: str,
    cache_dir: str,
    device: torch.device,
) -> Tuple[LayoutLMv3ForTokenClassification, LayoutLMv3Processor, int]:
    input_size = _read_input_size(checkpoint)
    try:
        processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,
            cache_dir=cache_dir,
        )
    except Exception:
        # Offline-safe fallback: build processor from local checkpoint assets.
        image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(checkpoint, local_files_only=True)
        processor = LayoutLMv3Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            apply_ocr=False,
        )
    processor.image_processor.size = {"height": input_size, "width": input_size}
    processor.image_processor.do_resize = True
    processor.image_processor.do_pad = True

    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    if hasattr(model, "layoutlmv3") and hasattr(model.layoutlmv3, "init_visual_bbox"):
        try:
            model.layoutlmv3.init_visual_bbox(image_size=(input_size // 16, input_size // 16))
        except Exception:
            pass
    return model, processor, input_size


@dataclass
class EncodedBatch:
    inputs: Dict[str, torch.Tensor]
    token_labels: List[List[int]]
    word_ids: List[List[Optional[int]]]
    true_word_labels: List[List[str]]
    sample_count: int


def _bucket_for_len(n_tokens: int, bounds: List[int]) -> int:
    for b in bounds:
        if n_tokens <= b:
            return b
    return bounds[-1]


def _collapse_true_labels(token_labels: List[int], word_ids: List[Optional[int]], label_list: List[str]) -> List[str]:
    prev = None
    out = []
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            lid = token_labels[pos]
            if lid != -100:
                out.append(label_list[lid])
            prev = wid
    return out


def _collapse_pred_labels(
    pred_ids: List[int],
    token_labels: List[int],
    word_ids: List[Optional[int]],
    label_list: List[str],
) -> List[str]:
    prev = None
    out = []
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid != prev:
            if token_labels[pos] != -100:
                pred_idx = pred_ids[pos] if pos < len(pred_ids) else 0
                out.append(label_list[pred_idx])
            prev = wid
    return out


def _prepare_batched_encodings(
    split_data,
    processor: LayoutLMv3Processor,
    input_size: int,
    label_list: List[str],
    batch_size: int = 4,
    max_length: int = 512,
    bucket_bounds: Optional[List[int]] = None,
    **kwargs
) -> Generator[EncodedBatch, None, None]:
    """Yield EncodedBatch objects one at a time to avoid holding all pixel_values in RAM."""
    bucket_bounds = bucket_bounds or DEFAULT_BUCKET_BOUNDS
    # Store only lightweight meta-data per sample; images are NOT stored here.
    buckets: Dict[int, List[Dict[str, Any]]] = {b: [] for b in bucket_bounds}

    def _encode_chunk(chunk: List[Dict[str, Any]]) -> EncodedBatch:
        # Convert PIL images inline and immediately discard refs
        images = [_pad_to_square(it["image"].convert("RGB"), input_size) for it in chunk]
        words = [it["words"] for it in chunk]
        boxes = [it["bboxes"] for it in chunk]
        labels = [it["ner_tags"] for it in chunk]

        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=labels,
            padding=kwargs.get("padding", "longest"),
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Free PIL images as soon as processor is done with them
        del images

        if "token_type_ids" not in encoding:
            encoding["token_type_ids"] = torch.zeros_like(encoding["input_ids"])

        word_ids = [encoding.word_ids(batch_index=i) for i in range(len(chunk))]
        token_labels = encoding["labels"].tolist()
        true_word_labels = [
            _collapse_true_labels(token_labels[i], word_ids[i], label_list)
            for i in range(len(chunk))
        ]

        return EncodedBatch(
            inputs={
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "bbox": encoding["bbox"],
                "pixel_values": encoding["pixel_values"],
                "token_type_ids": encoding["token_type_ids"],
            },
            token_labels=token_labels,
            word_ids=word_ids,
            true_word_labels=true_word_labels,
            sample_count=len(chunk),
        )

    def _flush_bucket(bucket_key: int):
        items = buckets[bucket_key]
        while len(items) >= batch_size:
            chunk = items[:batch_size]
            del items[:batch_size]
            yield _encode_chunk(chunk)

    def _flush_all():
        for k in bucket_bounds:
            items = buckets[k]
            while items:
                chunk = items[:batch_size]
                del items[:batch_size]
                yield _encode_chunk(chunk)

    for ex in split_data:
        # Only store the lightweight raw dataset row (not decoded images)
        tok_len = len(ex["words"])
        bucket = _bucket_for_len(tok_len, bucket_bounds)
        buckets[bucket].append(ex)
        yield from _flush_bucket(bucket)

    yield from _flush_all()


def _export_onnx(args) -> Dict[str, Any]:
    checkpoint = _resolve_path(args.checkpoint, DEFAULT_CHECKPOINT)
    cache_dir = _resolve_cache_dir(args.cache_dir)
    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)

    device = torch.device("cpu")  # deterministic ONNX export
    model, processor, input_size = _load_model_and_processor(checkpoint, cache_dir, device)
    wrapper = SparseSpatialLayoutLMv3Wrapper(model).eval()

    dataset = load_dataset("nielsr/funsd", cache_dir=cache_dir)
    ex = dataset["test"][0]
    image = _pad_to_square(ex["image"].convert("RGB"), input_size)

    enc = processor(
        image,
        ex["words"],
        boxes=ex["bboxes"],
        padding="max_length",
        truncation=True,
        max_length=args.seq_max,
        return_tensors="pt",
    )
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    input_ids = enc["input_ids"][:, : min(args.seq_max, enc["input_ids"].shape[1])]
    attention_mask = enc["attention_mask"][:, : input_ids.shape[1]]
    bbox = enc["bbox"][:, : input_ids.shape[1], :]
    token_type_ids = enc["token_type_ids"][:, : input_ids.shape[1]]
    pixel_values = enc["pixel_values"]

    output_path = os.path.join(artifacts_dir, "layoutlmv3_fp32.onnx")

    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "bbox": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
            "pixel_values": {0: "batch"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )
    # Torch>=2.9 defaults to the new ONNX exporter which requires onnxscript.
    # Force legacy mode for broader environment compatibility.
    try:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, bbox, pixel_values, token_type_ids),
            output_path,
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, bbox, pixel_values, token_type_ids),
            output_path,
            **export_kwargs,
        )
    # Check model without keeping the full proto in RAM
    onnx.checker.check_model(output_path)

    return {
        "status": "ok",
        "onnx_fp32_path": output_path,
        "input_size": input_size,
        "opset": args.opset,
        "checkpoint": checkpoint,
    }


def _replace_bbox_consumers_with_sparse_path(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, int]:
    graph = model.graph
    input_names = [i.name for i in graph.input]
    if "bbox" not in input_names:
        raise RuntimeError("Sparse optimization requires input named 'bbox' in ONNX graph.")
    if "attention_mask" not in input_names:
        raise RuntimeError("Sparse optimization requires input named 'attention_mask' in ONNX graph.")

    original_consumers = [n for n in graph.node if "bbox" in list(n.input)]
    if not original_consumers:
        raise RuntimeError("Sparse optimization failed: no bbox consumers found to rewrite.")

    nz_raw = "sparse_bbox_nonzero_raw"
    nz_idx = "sparse_bbox_nonzero_idx"
    active_bbox = "sparse_bbox_active"
    bbox_shape = "sparse_bbox_shape"
    bbox_zero = "sparse_bbox_zero"
    bbox_sparse = "sparse_bbox_final"
    zero_tensor = onnx.helper.make_tensor(
        name="",
        data_type=onnx.TensorProto.INT64,
        dims=[1],
        vals=[0],
    )

    new_nodes = [
        onnx.helper.make_node("NonZero", inputs=["attention_mask"], outputs=[nz_raw], name="SparseNonZero"),
        onnx.helper.make_node(
            "Transpose",
            inputs=[nz_raw],
            outputs=[nz_idx],
            perm=[1, 0],
            name="SparseTransposeIndices",
        ),
        onnx.helper.make_node(
            "GatherND",
            inputs=["bbox", nz_idx],
            outputs=[active_bbox],
            name="SparseGatherActiveBbox",
        ),
        onnx.helper.make_node("Shape", inputs=["bbox"], outputs=[bbox_shape], name="SparseBboxShape"),
        onnx.helper.make_node(
            "ConstantOfShape",
            inputs=[bbox_shape],
            outputs=[bbox_zero],
            value=zero_tensor,
            name="SparseMakeZeroBbox",
        ),
        onnx.helper.make_node(
            "ScatterND",
            inputs=[bbox_zero, nz_idx, active_bbox],
            outputs=[bbox_sparse],
            name="SparseScatterActiveBbox",
        ),
    ]

    for idx, node in enumerate(new_nodes):
        graph.node.insert(idx, node)
    rewritten = 0
    for node in original_consumers:
        node.input[:] = [bbox_sparse if inp == "bbox" else inp for inp in node.input]
        rewritten += 1

    onnx.checker.check_model(model)
    return model, rewritten


def _optimize_sparse_spatial(args) -> Dict[str, Any]:
    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    src_path = args.onnx_in or os.path.join(artifacts_dir, "layoutlmv3_fp32.onnx")
    dst_path = args.onnx_out or os.path.join(artifacts_dir, "layoutlmv3_sparse.onnx")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source ONNX not found: {src_path}")

    model = onnx.load(src_path)
    model, rewritten = _replace_bbox_consumers_with_sparse_path(model)
    onnx.save(model, dst_path)

    return {
        "status": "ok",
        "onnx_sparse_path": dst_path,
        "source_onnx": src_path,
        "rewritten_bbox_consumers": rewritten,
    }


def _quantize_int8(args) -> Dict[str, Any]:
    ort_mod = _optional_import("onnxruntime")
    ort_quant_mod = _optional_import("onnxruntime.quantization")
    if ort_mod is None or ort_quant_mod is None:
        return {
            "status": "skipped",
            "reason": "onnxruntime and onnxruntime.quantization are required for INT8 quantization.",
        }

    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    src_path = args.onnx_in or os.path.join(artifacts_dir, "layoutlmv3_sparse.onnx")
    out_path = args.onnx_out or os.path.join(artifacts_dir, "layoutlmv3_sparse_int8.onnx")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"ONNX for quantization not found: {src_path}")

    quantize_dynamic = ort_quant_mod.quantize_dynamic
    QuantType = ort_quant_mod.QuantType

    try:
        quantize_dynamic(
            model_input=src_path,
            model_output=out_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=[
                "/roberta/embeddings/Gather",
                "/layoutlmv3/embeddings/word_embeddings/Gather",
                "/layoutlmv3/embeddings/position_embeddings/Gather",
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Dynamic Quantization failed: {e}")

    return {
        "status": "ok",
        "onnx_int8_path": out_path,
        "source_onnx": src_path,
        "calibration_method": "Dynamic (No Calibration Data)",
        "op_types_quantized": ["MatMul", "Gemm"],
    }


def _export_coreml(args) -> Dict[str, Any]:
    ct = _optional_import("coremltools")
    if ct is None:
        return {
            "status": "skipped",
            "reason": "coremltools is required for CoreML conversion.",
        }

    checkpoint = _resolve_path(args.checkpoint, DEFAULT_CHECKPOINT)
    cache_dir = _resolve_cache_dir(args.cache_dir)
    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = args.coreml_out or os.path.join(artifacts_dir, "layoutlmv3_sparse.mlpackage")

    model, processor, input_size = _load_model_and_processor(checkpoint, cache_dir, torch.device("cpu"))
    wrapper = SparseSpatialLayoutLMv3Wrapper(model).eval()

    dataset = load_dataset("nielsr/funsd", cache_dir=cache_dir)
    ex = dataset["test"][0]
    image = _pad_to_square(ex["image"].convert("RGB"), input_size)
    enc = processor(
        image,
        ex["words"],
        boxes=ex["bboxes"],
        padding="max_length",
        truncation=True,
        max_length=min(args.seq_max, 128),
        return_tensors="pt",
    )
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    example_inputs = (
        enc["input_ids"],
        enc["attention_mask"],
        enc["bbox"],
        enc["pixel_values"],
        enc["token_type_ids"],
    )

    traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    
    # ANE (Apple Neural Engine) prefers fixed shapes. 
    # Providing a RangeDim often drops the model to the GPU or CPU. 
    # Use fixed shapes for the primary benchmark sequence length.
    target_seq_len = int(args.seq_max)

    if args.use_image_input:
        # LayoutLMv3 (ImageNet) normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # CoreML ImageType normalization formula: output = (input * scale) + bias
        # Input is in [0, 255]. Standard normalization: (input/255.0 - mean) / std
        # => input * (1.0 / (255.0 * std)) - (mean / std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        scale = 1.0 / (255.0 * std)
        bias = -mean / std

        coreml_model = ct.convert(
            traced,
            source="pytorch",
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, target_seq_len), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, target_seq_len), dtype=np.int32),
                ct.TensorType(name="bbox", shape=(1, target_seq_len, 4), dtype=np.int32),
                ct.ImageType(
                    name="pixel_values",
                    shape=(1, 3, input_size, input_size),
                    scale=list(scale),
                    bias=list(bias),
                    color_layout=ct.colorlayout.RGB,
                ),
                ct.TensorType(name="token_type_ids", shape=(1, target_seq_len), dtype=np.int32),
            ],
            compute_units=ct.ComputeUnit.ALL,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
    else:
        coreml_model = ct.convert(
            traced,
            source="pytorch",
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, target_seq_len), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, target_seq_len), dtype=np.int32),
                ct.TensorType(name="bbox", shape=(1, target_seq_len, 4), dtype=np.int32),
                ct.TensorType(name="pixel_values", shape=(1, 3, input_size, input_size), dtype=np.float32),
                ct.TensorType(name="token_type_ids", shape=(1, target_seq_len), dtype=np.int32),
            ],
            compute_units=ct.ComputeUnit.ALL,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )
    coreml_model.save(out_path)

    return {
        "status": "ok",
        "coreml_path": out_path,
        "input_size": input_size,
    }


def _run_backend_torch(
    batches: Iterable[EncodedBatch],
    wrapper: SparseSpatialLayoutLMv3Wrapper,
    device: torch.device,
    label_list: List[str],
    warmup_runs: int,
    benchmark_runs: int,
) -> Dict[str, Any]:
    true_all: List[List[str]] = []
    pred_all: List[List[str]] = []
    total_ms = 0.0
    total_docs = 0

    wrapper = wrapper.to(device).eval()
    with torch.no_grad():
        for batch in batches:
            inputs = {k: v.to(device) for k, v in batch.inputs.items()}

            # Warmup: explicitly delete output to avoid hidden tensor accumulation
            for _ in range(max(0, warmup_runs)):
                _warmup_out = wrapper(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["bbox"],
                    inputs["pixel_values"],
                    inputs["token_type_ids"],
                )
                del _warmup_out

            logits = None
            t0 = time.perf_counter()
            for _ in range(max(1, benchmark_runs)):
                logits = wrapper(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    inputs["bbox"],
                    inputs["pixel_values"],
                    inputs["token_type_ids"],
                )
            dt_ms = (time.perf_counter() - t0) * 1000.0 / max(1, benchmark_runs)
            total_ms += dt_ms

            preds = logits.argmax(-1).detach().cpu().tolist() if logits is not None else []
            del logits, inputs  # free device tensors immediately
            for i in range(batch.sample_count):
                true_all.append(batch.true_word_labels[i])
                pred_all.append(
                    _collapse_pred_labels(
                        pred_ids=preds[i],
                        token_labels=batch.token_labels[i],
                        word_ids=batch.word_ids[i],
                        label_list=label_list,
                    )
                )
            total_docs += batch.sample_count

    return {
        "true": true_all,
        "pred": pred_all,
        "latency_ms_per_doc": float(total_ms / max(total_docs, 1)),
        "docs": total_docs,
    }


def _run_backend_onnx(
    onnx_path: str,
    batches: Iterable[EncodedBatch],
    label_list: List[str],
    warmup_runs: int,
    benchmark_runs: int,
    providers: List[str],
) -> Dict[str, Any]:
    ort_mod = _optional_import("onnxruntime")
    if ort_mod is None:
        return {"status": "unavailable", "reason": "onnxruntime is not installed."}
    if not os.path.exists(onnx_path):
        return {"status": "unavailable", "reason": f"ONNX file not found: {onnx_path}"}

    sess = ort_mod.InferenceSession(onnx_path, providers=providers)
    input_names = [i.name for i in sess.get_inputs()]
    active_providers = sess.get_providers()

    true_all: List[List[str]] = []
    pred_all: List[List[str]] = []
    total_ms = 0.0
    total_docs = 0

    for batch in batches:
        feed = {}
        for name in input_names:
            if name not in batch.inputs:
                continue
            feed[name] = batch.inputs[name].detach().cpu().numpy()

        # Warmup: delete output immediately
        for _ in range(max(0, warmup_runs)):
            _warmup_out = sess.run(None, feed)
            del _warmup_out

        t0 = time.perf_counter()
        outputs = None
        for _ in range(max(1, benchmark_runs)):
            outputs = sess.run(None, feed)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / max(1, benchmark_runs)
        total_ms += dt_ms
        del feed  # release numpy arrays from this batch

        logits = _select_logits_tensor(outputs or [], len(label_list)) if outputs else None
        del outputs
        preds = np.argmax(logits, axis=-1).tolist() if logits is not None else []
        if len(preds) != batch.sample_count:
            return {
                "status": "unavailable",
                "reason": f"Unexpected ONNX output batch size: got {len(preds)} expected {batch.sample_count}.",
            }
        for i in range(batch.sample_count):
            true_all.append(batch.true_word_labels[i])
            pred_all.append(
                _collapse_pred_labels(
                    pred_ids=preds[i],
                    token_labels=batch.token_labels[i],
                    word_ids=batch.word_ids[i],
                    label_list=label_list,
                )
            )
        total_docs += batch.sample_count

    return {
        "status": "ok",
        "true": true_all,
        "pred": pred_all,
        "latency_ms_per_doc": float(total_ms / max(total_docs, 1)),
        "docs": total_docs,
        "providers": active_providers,
    }


def _run_backend_coreml(
    coreml_path: str,
    batches: List[EncodedBatch],
    label_list: List[str],
    warmup_runs: int,
    benchmark_runs: int,
) -> Dict[str, Any]:
    ct = _optional_import("coremltools")
    if ct is None:
        return {"status": "unavailable", "reason": "coremltools is not installed."}
    if platform.system() != "Darwin":
        return {"status": "unavailable", "reason": "CoreML runtime is only available on macOS."}
    if not os.path.exists(coreml_path):
        return {"status": "unavailable", "reason": f"CoreML package not found: {coreml_path}"}

    try:
        # Override TMPDIR to circumvent Neural Engine compilation path permission bugs on mac
        original_tmp = os.environ.get("TMPDIR")
        safe_tmp = os.path.join(REPO_ROOT, ".hf_cache", "coreml_tmp")
        os.makedirs(safe_tmp, exist_ok=True)
        os.environ["TMPDIR"] = safe_tmp
        try:
            mlmodel = ct.models.MLModel(coreml_path, compute_units=ct.ComputeUnit.ALL)
        finally:
            if original_tmp is not None:
                os.environ["TMPDIR"] = original_tmp
            else:
                del os.environ["TMPDIR"]
    except Exception as e:
        return {"status": "unavailable", "reason": f"CoreML model load failed: {e}"}
    spec_inputs = [i.name for i in mlmodel.get_spec().description.input]

    true_all: List[List[str]] = []
    pred_all: List[List[str]] = []
    total_ms = 0.0
    total_docs = 0

    for batch in batches:
        for i in range(batch.sample_count):
            feed = {}
            for name in spec_inputs:
                if name not in batch.inputs:
                    continue
                arr = batch.inputs[name][i : i + 1].detach().cpu().numpy()
                feed[name] = arr.astype(np.float32) if name == "pixel_values" else arr.astype(np.int32)

            try:
                for _ in range(max(0, warmup_runs)):
                    _warmup_out = mlmodel.predict(feed)
                    del _warmup_out
            except Exception as e:
                return {"status": "unavailable", "reason": f"CoreML inference warmup failed: {e}"}

            output = None
            t0 = time.perf_counter()
            try:
                for _ in range(max(1, benchmark_runs)):
                    output = mlmodel.predict(feed)
            except Exception as e:
                return {"status": "unavailable", "reason": f"CoreML inference failed: {e}"}
            dt_ms = (time.perf_counter() - t0) * 1000.0 / max(1, benchmark_runs)
            total_ms += dt_ms
            del feed  # free numpy arrays for this sample immediately

            if output is None:
                continue
            # Build candidate list without duplicates: prefer named "logits" key,
            # then add remaining values. _select_logits_tensor deduplicates by id.
            logits_candidates: List[Any] = []
            if "logits" in output:
                logits_candidates.append(output["logits"])
            for k, v in output.items():
                if k != "logits":
                    logits_candidates.append(v)
            logits = _select_logits_tensor(logits_candidates, len(label_list))
            del output, logits_candidates
            if logits.ndim == 2:
                pred_ids = np.argmax(logits, axis=-1).tolist()
            else:
                pred_ids = np.argmax(logits, axis=-1).tolist()[0]
            if not isinstance(pred_ids, list):
                return {"status": "unavailable", "reason": "CoreML logits output shape is not token-class compatible."}
            true_all.append(batch.true_word_labels[i])
            pred_all.append(
                _collapse_pred_labels(
                    pred_ids=pred_ids,
                    token_labels=batch.token_labels[i],
                    word_ids=batch.word_ids[i],
                    label_list=label_list,
                )
            )
            total_docs += 1

    return {
        "status": "ok",
        "true": true_all,
        "pred": pred_all,
        "latency_ms_per_doc": float(total_ms / max(total_docs, 1)),
        "docs": total_docs,
    }


def _compute_f1(true_all: List[List[str]], pred_all: List[List[str]]) -> Dict[str, float]:
    try:
        from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    return {
        "precision": float(precision_score(true_all, pred_all)),
        "recall": float(recall_score(true_all, pred_all)),
        "f1": float(f1_score(true_all, pred_all)),
        "accuracy": float(accuracy_score(true_all, pred_all)),
    }


def _validate_onnx_dynamic_shapes(onnx_path: str) -> Dict[str, Any]:
    ort_mod = _optional_import("onnxruntime")
    if ort_mod is None:
        return {"status": "skipped", "reason": "onnxruntime is not installed."}
    if not os.path.exists(onnx_path):
        return {"status": "skipped", "reason": f"ONNX not found: {onnx_path}"}

    sess = ort_mod.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    lengths = [1, 128, 512]
    checks = {}
    for seq_len in lengths:
        feed = {
            "input_ids": np.ones((1, seq_len), dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "bbox": np.ones((1, seq_len, 4), dtype=np.int64),
            "token_type_ids": np.zeros((1, seq_len), dtype=np.int64),
            "pixel_values": np.zeros((1, 3, 384, 384), dtype=np.float32),
        }
        feed = {k: v for k, v in feed.items() if k in input_names}
        try:
            outputs = sess.run(None, feed)
            checks[str(seq_len)] = {"ok": True, "output_shape": list(outputs[0].shape)}
        except Exception as e:
            checks[str(seq_len)] = {"ok": False, "error": str(e)}
    return {"status": "ok", "checks": checks}


def _benchmark(args) -> Dict[str, Any]:
    checkpoint = _resolve_path(args.checkpoint, DEFAULT_CHECKPOINT)
    cache_dir = _resolve_cache_dir(args.cache_dir)
    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)

    torch_device = _pick_device(args.device)
    model, processor, input_size = _load_model_and_processor(checkpoint, cache_dir, torch_device)
    wrapper = SparseSpatialLayoutLMv3Wrapper(model).eval()

    dataset = load_dataset("nielsr/funsd", cache_dir=cache_dir)
    test_split = dataset["test"]
    eval_limit = getattr(args, "eval_limit", None)
    if eval_limit is not None:
        test_split = test_split.select(range(min(eval_limit, len(test_split))))
    label_list = dataset["train"].features["ner_tags"].feature.names

    def _make_batches():
        """Return a fresh generator each time — avoids keeping all batches in RAM."""
        return _prepare_batched_encodings(
            split_data=test_split,
            processor=processor,
            input_size=input_size,
            label_list=label_list,
            batch_size=args.batch_size,
            max_length=args.seq_max,
        )

    backends: Dict[str, Dict[str, Any]] = {}

    # ── PyTorch backend ──────────────────────────────────────────────────────
    torch_res = _run_backend_torch(
        batches=_make_batches(),
        wrapper=wrapper,
        device=torch_device,
        label_list=label_list,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    torch_metrics = _compute_f1(torch_res["true"], torch_res["pred"])
    backends["pytorch_fp32"] = {
        "status": "ok",
        "latency_ms_per_doc": torch_res["latency_ms_per_doc"],
        "metrics": torch_metrics,
    }
    # Free model from device memory before loading ONNX sessions
    wrapper.cpu()
    del wrapper, model
    _free_memory()

    ort_mod = _optional_import("onnxruntime")
    onnx_providers = _pick_onnx_providers(ort_mod, args.onnx_provider) if ort_mod is not None else ["CPUExecutionProvider"]

    # ── ONNX FP32 backend ───────────────────────────────────────────────────
    onnx_fp32 = getattr(args, "onnx_fp32", None) or os.path.join(artifacts_dir, "layoutlmv3_sparse.onnx")
    onnx_fp32_res = _run_backend_onnx(
        onnx_path=onnx_fp32,
        batches=_make_batches(),
        label_list=label_list,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        providers=onnx_providers,
    )
    if onnx_fp32_res.get("status") == "ok":
        backends["onnx_fp32"] = {
            "status": "ok",
            "latency_ms_per_doc": onnx_fp32_res["latency_ms_per_doc"],
            "metrics": _compute_f1(onnx_fp32_res["true"], onnx_fp32_res["pred"]),
            "providers": onnx_fp32_res.get("providers", onnx_providers),
        }
    else:
        backends["onnx_fp32"] = onnx_fp32_res
    del onnx_fp32_res
    _free_memory()

    # ── ONNX INT8 backend ───────────────────────────────────────────────────
    onnx_int8 = getattr(args, "onnx_int8", None) or os.path.join(artifacts_dir, "layoutlmv3_sparse_int8.onnx")
    onnx_int8_res = _run_backend_onnx(
        onnx_path=onnx_int8,
        batches=_make_batches(),
        label_list=label_list,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        providers=onnx_providers,
    )
    if onnx_int8_res.get("status") == "ok":
        backends["onnx_int8"] = {
            "status": "ok",
            "latency_ms_per_doc": onnx_int8_res["latency_ms_per_doc"],
            "metrics": _compute_f1(onnx_int8_res["true"], onnx_int8_res["pred"]),
            "providers": onnx_int8_res.get("providers", onnx_providers),
        }
    else:
        backends["onnx_int8"] = onnx_int8_res
    del onnx_int8_res
    _free_memory()

    # ── CoreML backend ───────────────────────────────────────────────────────
    coreml_path = getattr(args, "coreml_path", None) or os.path.join(artifacts_dir, "layoutlmv3_sparse.mlpackage")
    coreml_res = _run_backend_coreml(
        coreml_path=coreml_path,
        batches=_make_batches(),
        label_list=label_list,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    if coreml_res.get("status") == "ok":
        backends["coreml"] = {
            "status": "ok",
            "latency_ms_per_doc": coreml_res["latency_ms_per_doc"],
            "metrics": _compute_f1(coreml_res["true"], coreml_res["pred"]),
        }
    else:
        backends["coreml"] = coreml_res
    del coreml_res
    _free_memory()

    fp32_f1 = backends["pytorch_fp32"]["metrics"]["f1"]
    onnx_fp32_f1 = (
        backends["onnx_fp32"]["metrics"]["f1"]
        if backends.get("onnx_fp32", {}).get("status") == "ok"
        else fp32_f1
    )
    summary: Dict[str, Any] = {"quantization_f1_loss_abs": None, "speedups_vs_fp32": {}}
    if backends["onnx_int8"].get("status") == "ok":
        q_f1 = backends["onnx_int8"]["metrics"]["f1"]
        summary["quantization_f1_loss_abs"] = float(onnx_fp32_f1 - q_f1)
        summary["speedups_vs_fp32"]["onnx_int8"] = float(
            backends["pytorch_fp32"]["latency_ms_per_doc"] / max(backends["onnx_int8"]["latency_ms_per_doc"], 1e-9)
        )
        if backends["onnx_fp32"].get("status") == "ok":
            summary["speedups_vs_fp32"]["onnx_int8_vs_onnx_fp32"] = float(
                backends["onnx_fp32"]["latency_ms_per_doc"] / max(backends["onnx_int8"]["latency_ms_per_doc"], 1e-9)
            )
    if backends["coreml"].get("status") == "ok":
        summary["speedups_vs_fp32"]["coreml"] = float(
            backends["pytorch_fp32"]["latency_ms_per_doc"] / max(backends["coreml"]["latency_ms_per_doc"], 1e-9)
        )

    out = {
        "status": "ok",
        "latency_scope": "model_only",
        "docs_evaluated": sum(b.sample_count for b in batches),
        "backends": backends,
        "summary": summary,
    }
    if args.benchmark_json:
        with open(args.benchmark_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def _manual_simulator_steps(coreml_path: str) -> List[str]:
    return [
        "1. Install Xcode from the Mac App Store (not just Command Line Tools): https://apps.apple.com/app/xcode/id497799835",
        "2. Open Xcode and create or use an iOS app target (SwiftUI, any bundle ID).",
        f"3. Drag '{coreml_path}' into the Xcode project navigator. Tick 'Copy items if needed'.",
        "4. Add CoreMLInference.swift and ContentView.swift from LayoutLM/iOSTestApp/ to the target.",
        "5. Build and run on iPhone 15 (or later) Simulator.",
        "6. Press 'Run Inference'. Expected: 'Success! Output shape: var_1295: [1, 512, 7]'.",
        f"7. Once Xcode is installed, compile the model package manually with:\n"
        f"   xcrun coremlcompiler compile '{coreml_path}' <output_dir>",
    ]


def _is_xcode_app_installed() -> bool:
    """Return True only if a full Xcode.app (not just CLT) is present."""
    proc = subprocess.run(
        ["xcode-select", "-p"],
        capture_output=True,
        text=True,
    )
    path = proc.stdout.strip()
    # CLT path is typically /Library/Developer/CommandLineTools
    # Full Xcode path ends with Xcode.app/Contents/Developer
    return "Xcode.app" in path


def _sim_test(args) -> Dict[str, Any]:
    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    coreml_path = getattr(args, "coreml_path", None) or getattr(args, "coreml_out", None) or os.path.join(
        artifacts_dir, "layoutlmv3_sparse.mlpackage"
    )

    result = {
        "status": "skipped",
        "coreml_path": coreml_path,
        "compiled_path": None,
        "manual_steps": _manual_simulator_steps(coreml_path),
    }
    if not os.path.exists(coreml_path):
        result["reason"] = f"CoreML package not found: {coreml_path}"
        return result

    xcrun = shutil.which("xcrun")
    if not xcrun:
        result["reason"] = "xcrun not available. Install Xcode from the Mac App Store."
        return result

    # coremlcompiler is only available in a full Xcode.app, not Command Line Tools.
    if not _is_xcode_app_installed():
        result["reason"] = (
            "Only Xcode Command Line Tools are installed. "
            "'coremlcompiler' requires the full Xcode.app. "
            "Install Xcode from the Mac App Store: https://apps.apple.com/app/xcode/id497799835 "
            "then run: xcrun coremlcompiler compile "
            f"'{coreml_path}' <output_dir>"
        )
        return result

    compile_dir = os.path.join(artifacts_dir, "compiled_coreml")
    os.makedirs(compile_dir, exist_ok=True)
    cmd = [xcrun, "coremlcompiler", "compile", coreml_path, compile_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        result["reason"] = f"coremlcompiler failed: {proc.stderr.strip()}"
        return result

    result["status"] = "ok"
    result["compiled_path"] = compile_dir
    result["simctl_detected"] = bool(shutil.which("simctl"))
    result["reason"] = "CoreML package compiled successfully; run app-level inference on simulator with manual steps."
    return result


def _inspect_coreml(args) -> Dict[str, Any]:
    """Print CoreML model inputs, outputs, and key metadata."""
    ct = _optional_import("coremltools")
    if ct is None:
        return {"status": "skipped", "reason": "coremltools is required for inspection."}

    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    coreml_path = getattr(args, "coreml_path", None) or os.path.join(artifacts_dir, "layoutlmv3_sparse.mlpackage")

    if not os.path.exists(coreml_path):
        return {"status": "error", "reason": f"CoreML package not found: {coreml_path}"}

    try:
        mlmodel = ct.models.MLModel(coreml_path)
    except Exception as e:
        return {"status": "error", "reason": f"Failed to load model: {e}"}

    spec = mlmodel.get_spec()

    # Dtype code -> friendly name mapping (CoreML protobuf constants)
    _DTYPE_MAP = {
        65568: "float32",
        65600: "float64",
        131104: "int32",
        131136: "int64",
        196640: "uint8",
    }

    inputs = []
    for inp in spec.description.input:
        t = inp.type.multiArrayType
        inputs.append({
            "name": inp.name,
            "shape": list(t.shape),
            "dtype": _DTYPE_MAP.get(t.dataType, str(t.dataType)),
        })

    outputs = []
    for out in spec.description.output:
        t = out.type.multiArrayType
        outputs.append({
            "name": out.name,
            "shape": list(t.shape),
            "dtype": _DTYPE_MAP.get(t.dataType, str(t.dataType)),
        })

    return {
        "status": "ok",
        "coreml_path": coreml_path,
        "spec_version": spec.specificationVersion,
        "inputs": inputs,
        "outputs": outputs,
        "swift_note": (
            "Use these shapes in CoreMLInference.swift. "
            f"seqLen={inputs[0]['shape'][1] if inputs else '?'}, "
            f"imgSize={inputs[3]['shape'][2] if len(inputs) > 3 else '?'}, "
            f"output_name='{outputs[0]['name'] if outputs else '?'}'."
        ),
    }


def _run_all(args) -> Dict[str, Any]:
    artifacts_dir = _resolve_path(args.artifacts_dir, DEFAULT_ARTIFACTS_DIR)
    os.makedirs(artifacts_dir, exist_ok=True)
    report: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": _resolve_path(args.checkpoint, DEFAULT_CHECKPOINT),
        "artifacts_dir": artifacts_dir,
        "steps": {},
    }

    export_res = _export_onnx(args)
    report["steps"]["export_onnx"] = export_res

    sparse_res = _optimize_sparse_spatial(args)
    report["steps"]["optimize_sparse_spatial"] = sparse_res

    quant_res = _quantize_int8(args)
    report["steps"]["quantize_int8"] = quant_res

    coreml_res = _export_coreml(args)
    report["steps"]["export_coreml"] = coreml_res

    benchmark_json = os.path.join(artifacts_dir, "benchmark_report.json")
    setattr(args, "benchmark_json", benchmark_json)
    bench_res = _benchmark(args)
    report["steps"]["benchmark"] = bench_res

    sim_json = os.path.join(artifacts_dir, "simulator_report.json")
    sim_res = _sim_test(args)
    report["steps"]["sim_test"] = sim_res

    validate_path = sparse_res.get("onnx_sparse_path", os.path.join(artifacts_dir, "layoutlmv3_sparse.onnx"))
    report["steps"]["onnx_dynamic_validation"] = _validate_onnx_dynamic_shapes(validate_path)

    final_report_path = os.path.join(artifacts_dir, "onnx_coreml_run_all_report.json")
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(sim_json, "w", encoding="utf-8") as f:
        json.dump(sim_res, f, indent=2)
    return {"status": "ok", "report_path": final_report_path, "report": report}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated ONNX/CoreML pipeline for LayoutLMv3 experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to LayoutLMv3 checkpoint")
        p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="HF cache dir")
        p.add_argument("--artifacts_dir", type=str, default=DEFAULT_ARTIFACTS_DIR, help="Output artifacts dir")
        p.add_argument("--seq_min", type=int, default=1, help="Minimum dynamic seq length")
        p.add_argument("--seq_max", type=int, default=512, help="Maximum dynamic seq length")
        p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Execution device")

    p_export = sub.add_parser("export_onnx", help="Export sparse-wrapper LayoutLMv3 to ONNX")
    add_common(p_export)
    p_export.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")

    p_sparse = sub.add_parser("optimize_sparse_spatial", help="Apply ONNX sparse bbox gather/scatter graph rewrite")
    add_common(p_sparse)
    p_sparse.add_argument("--onnx_in", type=str, default=None, help="Input ONNX path")
    p_sparse.add_argument("--onnx_out", type=str, default=None, help="Output sparse ONNX path")

    p_quant = sub.add_parser("quantize_int8", help="INT8 static quantization with ORT calibration")
    add_common(p_quant)
    p_quant.add_argument("--onnx_in", type=str, default=None, help="Input ONNX path")
    p_quant.add_argument("--onnx_out", type=str, default=None, help="Output INT8 ONNX path")


    p_coreml = sub.add_parser("export_coreml", help="Export CoreML package from sparse wrapper")
    add_common(p_coreml)
    p_coreml.add_argument("--coreml_out", type=str, default=None, help="Output .mlpackage path")
    p_coreml.add_argument("--use_image_input", action="store_true", help="Use ImageType for pixel_values")

    p_bench = sub.add_parser("benchmark", help="Benchmark fp32 vs onnx int8 vs coreml")
    add_common(p_bench)
    p_bench.add_argument("--onnx_fp32", type=str, default=None, help="Path to fp32 ONNX model")
    p_bench.add_argument("--onnx_int8", type=str, default=None, help="Path to INT8 ONNX model")
    p_bench.add_argument(
        "--onnx_provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "coreml", "cuda"],
        help="ONNX Runtime provider selection policy",
    )
    p_bench.add_argument("--coreml_path", type=str, default=None, help="Path to CoreML .mlpackage")
    p_bench.add_argument("--eval_limit", type=int, default=None, help="Optional eval doc cap")
    p_bench.add_argument("--batch_size", type=int, default=4, help="Tokenization batch size")
    p_bench.add_argument("--warmup_runs", type=int, default=2, help="Warmup runs per batch")
    p_bench.add_argument("--benchmark_runs", type=int, default=5, help="Timed runs per batch")
    p_bench.add_argument("--confidence_report", action="store_true", help="Include confidence-style summary fields")
    p_bench.add_argument("--benchmark_json", type=str, default=None, help="Optional benchmark report JSON path")

    p_sim = sub.add_parser("sim_test", help="Compile/check CoreML package and provide simulator instructions")
    add_common(p_sim)
    p_sim.add_argument("--coreml_path", type=str, default=None, help="Path to CoreML .mlpackage")

    p_inspect = sub.add_parser("inspect_coreml", help="Print CoreML model inputs, outputs, and metadata")
    add_common(p_inspect)
    p_inspect.add_argument("--coreml_path", type=str, default=None, help="Path to CoreML .mlpackage")

    p_all = sub.add_parser("run_all", help="Run full ONNX/CoreML pipeline")
    add_common(p_all)
    p_all.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")

    p_all.add_argument("--eval_limit", type=int, default=None, help="Optional eval doc cap")
    p_all.add_argument("--batch_size", type=int, default=4, help="Tokenization batch size")
    p_all.add_argument("--warmup_runs", type=int, default=2, help="Warmup runs per batch")
    p_all.add_argument("--benchmark_runs", type=int, default=5, help="Timed runs per batch")
    p_all.add_argument("--confidence_report", action="store_true", help="Include confidence-style summary fields")
    p_all.add_argument("--onnx_in", type=str, default=None, help="Override sparse optimization input ONNX")
    p_all.add_argument("--onnx_out", type=str, default=None, help="Override sparse optimization output ONNX")
    p_all.add_argument("--onnx_fp32", type=str, default=None, help="Override benchmark fp32 ONNX path")
    p_all.add_argument(
        "--onnx_provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "coreml", "cuda"],
        help="ONNX Runtime provider selection policy",
    )
    p_all.add_argument("--coreml_out", type=str, default=None, help="Override output .mlpackage path")
    p_all.add_argument("--use_image_input", action="store_true", help="Use ImageType for pixel_values")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "export_onnx":
        out = _export_onnx(args)
    elif args.command == "optimize_sparse_spatial":
        out = _optimize_sparse_spatial(args)
    elif args.command == "quantize_int8":
        out = _quantize_int8(args)
    elif args.command == "export_coreml":
        out = _export_coreml(args)
    elif args.command == "benchmark":
        out = _benchmark(args)
    elif args.command == "sim_test":
        out = _sim_test(args)
    elif args.command == "inspect_coreml":
        out = _inspect_coreml(args)
    elif args.command == "run_all":
        out = _run_all(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
