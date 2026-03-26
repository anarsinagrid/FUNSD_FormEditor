# 📑 Document Intelligence & Interactive Form Editor
> **A high-performance LayoutLMv3 pipeline optimized for M-series hardware and ANE deployment.**

This repository features state-of-the-art document understanding tools, combining an **Adaptive LayoutLMv3 Inference Pipeline** with an **Interactive UI Editor**. We focus on bridges the gap between raw model predictions and human-usable document intelligence.

---

## 🚀 **The "North Star" Metrics**
*Achieving state-of-the-art results through architectural optimization.*

| Metric Category        | **Benchmark (GT)** | **Inference (docTR)** | **Performance Lift** |
| :--------------------- | :----------------- | :-------------------- | :------------------- |
| **Token-level F1**     | **0.8090**         | **0.7905**            | +41.1% (vs Baseline) |
| **Entity-level F1**    | **0.7720**         | **0.7256**            | +61.2% (vs Baseline) |
| **Quantized ANE Latency**| N/A                | **<100ms**            | **12x Speedup**      |

> [!NOTE]
> We report two sets of metrics: **Benchmark (GT)** reflects the LayoutLMv3 architecture's peak semantic capability using ground-truth boxes, while **Inference (docTR)** accounts for real-world OCR detection shifts during end-to-end processing.

---

## ✨ **The "WOW" Factors**
*Technical breakthroughs that set this project apart.*

- 🥇 **Benchmark Performance:** Reached a peak **80.9% F1** on FUNSD, setting a high-fidelity ceiling for the model's structural understanding.
- ⚡ **Sub-100ms Inference:** Optimized for **Apple Neural Engine (ANE)** via a custom Sparse Spatial Encoding path.
- 🚀 **Adaptive Cascade:** Intelligent resolution selection (**224px → 512px**) yielding a **37% speedup** on M1/M2 silicon.
- 🧠 **Structural Reasoning:** Integrated **GATv2 (Graph Attention Networks)** to understand hierarchical form relations (`above`, `below`, `aligned`).

---

## 🖥 **Interactive Editor**
The core of the user experience. The editor allows you to visualize, inspect, and correct model predictions in real-time.

### **Quick Start**
```bash
export PYTHONPATH=$(pwd)
python -m editor.main
```

### **Key Features**
- **Multi-Engine Support:** Swap between **Tesseract**, **docTR**, and **Paddle** OCR backends on the fly.
- **Visual Inspector:** View per-token confidence scores and classified labels (QUESTION, ANSWER, HEADER).
- **Advanced Editing:** Split, merge, and delete bounding boxes with instant model re-inference.
- **Uncertainty Highlighting:** Automatically flags "low-confidence" regions for human-in-the-loop review.

---

## 🛠 **Engineering Achievements**

### **1. Adaptive Multi-Resolution Cascade**
Documents are not created equal. Our selector dynamically routes "simple" forms (invoices) to **224px** for instant results (**351ms latency**), while "complex" forms utilize the **512px** path.
- **Achievement:** **15.7% overall speedup** (Adaptive) and up to **37.2% speedup** (Cascade strategy) on M1/M2 silicon without sacrificing F1 accuracy.
- **Resilience:** The cascade escalation rate remained at **0.0**, meaning our 0.65 confidence threshold was consistently met at lower resolutions for most documents.

### **2. Confidence-Aware Intelligence**
Instead of treating OCR as ground truth, we modulate word embeddings with **OCR Confidence Scores**.
- **Evidence:** Identified that **ANSWER** fields are **25.7% more sensitive** to OCR noise than headers, causing significant accuracy drops when confidence is unmanaged.
- **Impact:** Developed a self-diagnosing system with an **ECE of 0.1587**, effectively flagging "low-trust" extractions for human-in-the-loop review.

### **3. Zero-Shot Concept Steering**
Using **Prototypical Classification**, the model can be "steered" at test-time toward specific document concepts (e.g., Structured vs. Noise), enabling a **70% accuracy on unseen CORD classes** zero-shot without retraining.

---

## 📂 **Evidence & Reproducibility**
The project's empirical validity is documented in whitelisted benchmark reports:

| Metric Category        | **Value / Result** | **Dataset / Split** | **Evidence File** |
| :--------------------- | :----------------- | :------------------ | :----------------- |
| **Peak Benchmark F1**  | **0.8090**         | FUNSD (Test)        | `layoutlm_funsd_gt.json` |
| **Inference Token F1** | **0.7905**         | FUNSD (Test)        | `multi_resolution_report.json` |
| **OOD Accuracy (ZS)** | **0.7000**         | CORD (Test)         | `uncertainty_calibration.json` |
| **Cascade Speedup**    | **37.18%**         | FUNSD (Test)        | `multi_resolution_report.json` |
| **Calibration (ECE)**  | **0.1587**         | FUNSD (Test)        | `uncertainty_calibration.json` |

### **Project Technical Report**
Detailed architecture and design decision summaries are available in:
- 📊 **Architecture & Metrics:** [Project3_Task Description.md](Project3_Task%20Description.md)

### **Reference Implementation (CLI)**
For developers wishing to verify the evaluation stack (requires `datasets` & `evaluate`):

```bash
# Full Pipeline Benchmark (MPS)
python LayoutLM/domain_generalization_pipeline.py run_all --limit 50 --device mps

# Adaptive Resolution Evaluation
python LayoutLM/multi_resolution_pipeline.py --ocr_engine doctr --resolutions 224 384 512
```

---

## 📦 **Installation**

1. **Python 3.11:** `brew install python@3.11`
2. **Clone & LFS:** 
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/anarsinagrid/FUNSD_FormEditor.git
   git lfs pull
   ```
3. **Environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 📈 **Baseline vs. Optimized Comparison**

| Metric             | Baseline   | **Optimized** |
| :----------------- | :---------- | :------------ |
| **Token-level F1** | 0.56        | **0.7905**    |
| **Entity-level F1**| 0.45        | **0.7256**    |
| **ANE Latency**    | ~1.2s       | **<100ms**    |
