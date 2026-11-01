# Assignment 1 — Environment Setup & Execution Guide

This document provides instructions for reproducing all experiments for **DSAI 582: Agentic AI — Assignment 1** using `Qwen/Qwen2.5-0.5B-Instruct` on the **GSM8K** dataset.

The code can be executed either on a **free Google Colab T4 GPU instance** or **locally on CPU/GPU**.

---

## Option A: Google Colab Setup (Recommended for Fastest Execution)

### 1. Launch a New Notebook
1. Navigate to [colab.research.google.com](https://colab.research.google.com) and click **New Notebook**.
2. Select **Runtime → Change runtime type** from the top menu.
3. Under **Hardware accelerator**, select **T4 GPU** (available on the free tier) and save.

### 2. Upload Assignment Files
From the left sidebar, click the **Files** folder icon and upload:
- `assignment1.py`
- (Optional) `assignment1_results.json` if comparing existing outputs.

### 3. Install Dependencies
In the first code cell, execute:
```python
!pip install -q transformers datasets accelerate torch sentencepiece protobuf
```

### 4. Execute the Benchmark Suite
In the second cell, run:
```python
!python assignment1.py
```
*Alternatively, you can open and run `assignment1.py` cell-by-cell in Colab.*

---

## Option B: Local Execution (CPU or CUDA GPU)

### 1. Prerequisites
- Python 3.10+ installed
- At least 4 GB of available system RAM

### 2. Install Required Packages
```bash
pip install torch transformers datasets accelerate sentencepiece protobuf
```

### 3. Run Experiments
```bash
python assignment1.py
```

---

## Estimated Execution Times

| Evaluation Method | Problem Count | T4 GPU (Colab) | Local CPU |
| :--- | :---: | :---: | :---: |
| **Zero-Shot Prompting** | 30 questions | ~3–5 mins | ~15 mins |
| **Four-Shot Prompting** | 30 questions | ~4–6 mins | ~15 mins |
| **Self-Consistency (4 samples, $T=0.7$)** | 30 questions ($\times 4$ paths) | ~12–15 mins | ~45 mins |
| **Reflection / Revision (2-turn)** | 30 questions ($\times 2$ passes) | ~6–8 mins | ~20 mins |
| **Tool-Using Agent (Safe AST Calculator)** | 30 questions | ~5–8 mins | ~15 mins |
| **Total Benchmark Runtime** | **All 5 Methods** | **~30–45 mins** | **~1.5–2 hours** |

---

## Output Artifacts & Results

Upon script completion:
1. **Console Output:** Formatted summary tables displaying Accuracy (%), Total Generated Tokens, and Total Model Calls for all methods.
2. **`assignment1_results.json`:** Comprehensive JSON record containing:
   - Summary statistics for Question 1 and Question 2.
   - Complete reasoning trajectories, predicted answers, ground-truth references, token counts, and accuracy flags for every test problem.

---

## Compliance with Assignment Constraints

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` loaded directly via Hugging Face Transformers.
- **Frozen Weights:** All model parameters are explicitly frozen (`param.requires_grad = False`).
- **No Paid APIs:** Pure open-source local inference; zero proprietary API dependencies.
- **Concealed Solutions:** Reference solutions are strictly withheld from prompt contexts and used exclusively at evaluation time (`match(pred, ref)`).
- **Restricted Tool Execution:** The arithmetic calculator strictly uses Abstract Syntax Tree (`ast.parse`) evaluation with a whitelist of safe operators and mathematical functions. Unrestricted `eval()` and `exec()` are completely prohibited.
- **Dataset Partition:** GSM8K `train[0:10]` for development, `train[10:14]` for 4-shot demonstrations, and `test[0:30]` for held-out evaluation.
