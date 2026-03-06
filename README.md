# Form Generation & Document Understanding

This repository contains tools for evaluating Optical Character Recognition (OCR) engines and LayoutLM models on document datasets (like FUNSD). It also features an interactive UI Editor for visualizing, inspecting, and correcting model predictions.

## Project Structure

- **`LayoutLM/`**: Contains scripts to evaluate OCR engines and LayoutLMv3 models.
  - `eval_ocr_all.py`: Evaluates Tesseract, PaddleOCR, and docTR on the dataset and produces a comparison report.
  - `eval_layoutlm_all.py`: Evaluates LayoutLMv3 models coupled with different OCR backends.
  - `layoutlm_customOCR.py`: Wrappers and utilities tying different OCR engines to the evaluation pipeline.
- **`editor/`**: A PySide6-based graphical user interface for visualizing documents, bounding boxes, OCR results, and model classifications. It supports interactive editing and graph-linking predictions.
- **`requirements.txt`**: Python dependencies required to run the codebase.

## Setup & Installation

### Python Version

This project is tested with `Python 3.11.14`. Use Python `3.11.x` to avoid dependency incompatibilities.

1. Create a virtual environment (optional but recommended):
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```
   On Windows (PowerShell):
   ```bash
   py -3.11 -m venv venv
   .\\venv\\Scripts\\Activate.ps1
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   _(Note: Ensure you have system-level dependencies for Tesseract or PaddleOCR if you plan to use those specific backends)._

### Model Checkpoints (Git LFS)

Some model checkpoints in this repo are stored with Git LFS. After cloning, make sure you have the real weight files:
```bash
git lfs install
git lfs pull
```


## UI Navigation: Interactive Editor

The editor provides a user-friendly interface to visually inspect the LayoutLM predictions, OCR text, and bounding boxes, ensuring results are understandable and adjustable for peer evaluation.

### Running the Editor

```bash
export PYTHONPATH=$(pwd)
python editor/main.py
```

### Navigating the Editor UI

- **Opening a Document**: Use the **Open Image** button in the toolbar to load a scanned form or document image.
- **Zoom & Pan**:
  - Zoom in/out by holding **Ctrl + Scroll** (or equivalent on macOS).
  - Pan across the image by clicking and dragging or just using the scroll wheel.
- **Selecting Models & OCR**: Use the dropdown menus in the toolbar to select the desired OCR Engine (e.g., Tesseract, docTR, Paddle) and LayoutLM Model. This will re-run extraction on the document.
- **Editing Mode**:
  - Toggle the **Edit Mode** checkbox.
  - Click on bounding boxes to select them.
  - You can split, merge, or delete selected bounding boxes using the respective buttons in the toolbar.
- **Inspector Panel**: When you select a block, the Inspector on the right shows its OCR text, confidence score, and classified label (e.g., QUESTION, ANSWER, HEADER, OTHERS).


## Code Reproducibility: Running Evaluations

To replicate the evaluation scores for the OCR engines or the LayoutLM representations, you can run the provided scripts in the `LayoutLM` directory.

### 1. OCR Evaluation

Evaluate all configured OCR engines (Tesseract, PaddleOCR, docTR) and generate a side-by-side comparison report:

```bash
export PYTHONPATH=$(pwd)
python LayoutLM/eval_ocr_all.py
```

This script will output metrics like Character Error Rate (CER), Word Error Rate (WER), and Intersection over Union (IoU) for bounding boxes.

### 2. LayoutLM Evaluation

Evaluate the trained LayoutLMv3 models combined with different OCR backends:

```bash
export PYTHONPATH=$(pwd)
python LayoutLM/eval_layoutlm_all.py
```

This evaluates the token classification performance (e.g., F1 scores for HEADER, QUESTION, ANSWER classes) across the test split.

# Experiments & Results (From JSON Artifacts)

## Quick Comparison (3 Models)

| Model | OCR | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- |
| LayoutLMv3 (GT bboxes) `layoutlmv3-funsd` | gt | 0.8090 | 0.7844 | 0.8353 | 0.8290 |
| LayoutLMv3 (docTR-trained) `layoutlmv3-funsd-doctr` | doctr | 0.7279 | 0.7230 | 0.7330 | 0.7820 |
| UDOP `microsoft/udop-large` | (no OCR, processor `apply_ocr=False`) | 0.5473 | 0.4757 | 0.6442 | 0.6465 |

## OCR Evaluation (FUNSD)

Metrics: CER/WER lower is better; Word Recall/IoU/Order Consistency higher is better.

| Engine | CER | WER | Word Recall | Mean IoU | Median IoU | % IoU > 0.7 | Order Consistency | OCR/GT | GT Words | OCR Words | Docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| doctr | 0.1360 | 0.3303 | 0.8846 | 0.7195 | 0.7448 | 0.6306 | 0.9967 | 0.9620 | 30595 | 29432 | 199 |
| paddle | 0.2940 | 0.5500 | 0.2177 | 0.4958 | 0.4594 | 0.2355 | 0.9961 | 0.3320 | 30595 | 10158 | 199 |
| paddle-v4 | 0.2940 | 0.5500 | 0.2177 | 0.4958 | 0.4594 | 0.2355 | 0.9961 | 0.3320 | 30595 | 10158 | 199 |
| tesseract | 0.2998 | 0.5388 | 0.6997 | 0.5695 | 0.5847 | 0.2184 | 0.9961 | 0.8614 | 30595 | 26354 | 199 |

## LayoutLM Experiments (FUNSD Test Split)

Each row corresponds to a `LayoutLM/eval_results/layoutlm_*.json` artifact (also included in `layoutlm_comparison.json`).

| Model Dir | OCR Engine | F1 | Precision | Recall | Accuracy | Docs | GT Words | OCR Words | Time (s) | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layoutlmv3-funsd | doctr | 0.6780 | 0.6707 | 0.6854 | 0.7746 | 50 | 8707 | 8366 | 17.48 | `layoutlmv3-funsd/checkpoint-646` |
| layoutlmv3-funsd | gt | 0.8090 | 0.7844 | 0.8353 | 0.8290 | 50 | 8707 | 8707 | 4.24 | `layoutlmv3-funsd/checkpoint-646` |
| layoutlmv3-funsd | paddle | 0.6614 | 0.6033 | 0.7319 | 0.5640 | 50 | 8707 | 2545 | 218.30 | `layoutlmv3-funsd/checkpoint-646` |
| layoutlmv3-funsd | tesseract | 0.5606 | 0.5377 | 0.5856 | 0.6586 | 50 | 8707 | 7156 | 20.58 | `layoutlmv3-funsd/checkpoint-646` |
| layoutlmv3-funsd-doctr | doctr | 0.7279 | 0.7230 | 0.7330 | 0.7820 | 50 | 8707 | 8366 | 22.17 | `layoutlmv3-funsd-doctr/checkpoint-1250` |
| layoutlmv3-funsd-doctr-large | doctr | 0.7235 | 0.7069 | 0.7409 | 0.7794 | 50 | 8707 | 8366 | 34.92 | `layoutlmv3-funsd-doctr-large/checkpoint-1142` |
| layoutlmv3-funsd-paddle | paddle | 0.7639 | 0.7735 | 0.7545 | 0.7620 | 50 | 8707 | 2545 | 219.39 | `layoutlmv3-funsd-paddle/checkpoint-646` |
| layoutlmv3-funsd-tesseract | tesseract | 0.6545 | 0.6520 | 0.6570 | 0.7360 | 50 | 8707 | 7156 | 20.55 | `layoutlmv3-funsd-tesseract/checkpoint-1064` |
