# Form Generation & Document Understanding

This repository contains tools for evaluating Optical Character Recognition (OCR) engines and LayoutLM models on document datasets (like FUNSD). It also features an interactive UI Editor for visualizing, inspecting, and correcting model predictions.

## Setup & Installation

The primary focus of this repository is the Interactive Editor. Some model checkpoints in this repo are stored with Git LFS. **Since the Hugging Face model cache is tracked via LFS, it will download large files on the first run, which will take some time.**

To avoid future issues and download LFS files properly, follow these steps strictly:

1. **Install Python 3.11** (to avoid future dependency issues):

   ```bash
   brew install python@3.11
   ```

2. **Clone the repository without aggressively downloading LFS files upfront**:

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/anarsinagrid/FUNSD_FormEditor.git

   cd FUNSD_FormEditor
   ```

3. **Pull the large files via LFS**:

   ```bash
   git lfs pull
   ```

4. **Create a virtual environment** using the newly installed Python 3.11:

   ```bash
   python3.11 -m venv venv
   ```

5. **Activate the virtual environment**:

   ```bash
   source venv/bin/activate
   ```

   _(On Windows PowerShell: `.\\venv\\Scripts\\Activate.ps1`)_

6. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   _(Note: Ensure you have system-level dependencies for Tesseract or PaddleOCR if you plan to use those specific backends)._

---

## UI Navigation: Interactive Editor

The editor provides a user-friendly interface to visually inspect the LayoutLM predictions, OCR text, and bounding boxes, ensuring results are understandable and adjustable.

### Running the Editor

```bash
export PYTHONPATH=$(pwd)
python -m editor.main
```

> **Note:** The first time you run this, `transformers` might download its required model files (like LayoutLM architecture weights). Depending on your internet speed, the CLI might appear momentarily frozen—this is normal.

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

---

## Code Reproducibility: Running Evaluations

To replicate the evaluation scores for the OCR engines or the LayoutLM representations, you can run the provided scripts in the `LayoutLM` directory.

> **Important:** The trained model checkpoints are **not available on GitHub** to save repository space. You will need to train the models yourself to generate these checkpoints before you can run the LayoutLM evaluations.
>
> If you simply want to view the detailed metric results without running the code, full JSON reports are available directly in the [`LayoutLM/eval_results/`](LayoutLM/eval_results/) directory.

### 1. OCR Evaluation

Evaluate all configured OCR engines (Tesseract, PaddleOCR, docTR) and generate a side-by-side comparison report:

```bash
export PYTHONPATH=$(pwd)
python LayoutLM/eval_ocr_all.py
```

### 2. LayoutLM Evaluation

Evaluate the trained LayoutLMv3 models combined with different OCR backends:

```bash
export PYTHONPATH=$(pwd)
python LayoutLM/eval_layoutlm_all.py
```

---

# Experiments & Results

## OCR Evaluation (FUNSD)

Metrics: CER/WER lower is better; Word Recall/Mean IoU higher is better.

| Engine             | CER        | WER        | Word Recall | Mean IoU   |
| ------------------ | ---------- | ---------- | ----------- | ---------- |
| **doctr**          | **0.1360** | **0.3303** | **0.8846**  | **0.7195** |
| paddle / paddle-v4 | 0.2940     | 0.5500     | 0.2177      | 0.4958     |
| tesseract          | 0.2998     | 0.5388     | 0.6997      | 0.5695     |

## LayoutLM Experiments (FUNSD Test Split)

Sorted by F1 Score. Note: `gt` uses ground-truth bounding boxes instead of an OCR engine.

| Model Dir                    | OCR Engine | F1         | Precision | Recall | Accuracy |
| ---------------------------- | ---------- | ---------- | --------- | ------ | -------- |
| layoutlmv3-funsd             | gt         | **0.8090** | 0.7844    | 0.8353 | 0.8290   |
| layoutlmv3-funsd-paddle      | paddle     | 0.7639     | 0.7735    | 0.7545 | 0.7620   |
| layoutlmv3-funsd-doctr       | doctr      | 0.7279     | 0.7230    | 0.7330 | 0.7820   |
| layoutlmv3-funsd-doctr-large | doctr      | 0.7235     | 0.7069    | 0.7409 | 0.7794   |
| layoutlmv3-funsd             | doctr      | 0.6780     | 0.6707    | 0.6854 | 0.7746   |
| layoutlmv3-funsd             | paddle     | 0.6614     | 0.6033    | 0.7319 | 0.5640   |
| layoutlmv3-funsd-tesseract   | tesseract  | 0.6545     | 0.6520    | 0.6570 | 0.7360   |
| layoutlmv3-funsd             | tesseract  | 0.5606     | 0.5377    | 0.5856 | 0.6586   |
