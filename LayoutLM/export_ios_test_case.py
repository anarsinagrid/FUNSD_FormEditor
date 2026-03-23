import os
import json
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import LayoutLMv3Processor

def export_test_case(doc_idx=0, output_dir="LayoutLM/iOSTestApp/TestData"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading FUNSD test dataset...")
    dataset = load_dataset("nielsr/funsd")
    example = dataset["test"][doc_idx]
    
    # Processor setup (matches onnx_coreml_pipeline.py)
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False # Use boxes from dataset directly
    )
    
    image = example["image"].convert("RGB")
    words = example["words"]
    boxes = example["bboxes"]
    
    print(f"Processing document {doc_idx}...")
    encoding = processor(
        image,
        words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Prepare JSON data
    # CoreML expects int32 for these tensors in our export
    test_data = {
        "input_ids": encoding["input_ids"].squeeze(0).tolist(),
        "attention_mask": encoding["attention_mask"].squeeze(0).tolist(),
        "bbox": encoding["bbox"].squeeze(0).tolist(),
        "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0).tolist(),
        "image_size": [image.size[1], image.size[0]], # height, width
        "doc_idx": doc_idx,
        "words": words
    }
    
    json_path = os.path.join(output_dir, f"test_case_{doc_idx}.json")
    with open(json_path, "w") as f:
        json.dump(test_data, f)
    
    img_path = os.path.join(output_dir, f"test_case_{doc_idx}.png")
    image.save(img_path)
    
    print(f"Successfully exported test case to:")
    print(f"  - {json_path}")
    print(f"  - {img_path}")

if __name__ == "__main__":
    export_test_case()
