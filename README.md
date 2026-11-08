# 🚀 Gradient-Free AttnRes for LLMs: Mitigating Depth Dilution via CMA-ES

**Author:** Zhanjiang Wang  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://org.licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E)](https://huggingface.co/)

> **Lightweight, Zero-Order Architectural Optimization for PreNorm Transformers.**
> 
> *A 112-second evolutionary search (CMA-ES) over a 24-dimensional residual routing space reduces native GPT-2 validation Loss from **9.6361** to **7.3976** ($\Delta = 2.2384$) while completely freezing pre-trained backbone parameters.*

---

## 📌 Abstract

To address the **Depth Dilution Problem** in deep Transformer architectures under the Pre-Layer Normalization (Pre-LN) paradigm, this project implements a lightweight, derivative-free optimization framework using the Covariance Matrix Adaptation Evolution Strategy (**CMA-ES**). 

Inspired by Moonshot AI's **Attention Residuals (AttnRes)** conceptual model (*arXiv:2603.15031*), we search for layer-wise dynamic scaling factors ($\alpha$) and routing parameters on a GPT-2 Small (124M, 12 layers) backbone. Without backpropagation or modifying pre-trained weight matrices, CMA-ES automatically discovers an asymmetric **"dumbbell-shaped"** residual routing topology, prioritizing early features (Layers 02–03) and final output logit construction (Layer 12, holding 17.9% historical weight share).

---

## 📊 Experimental Results

### 1. Convergence Trajectory (15 Generations)

| Generation | Best Loss | Loss Delta ($\Delta$) | Evolutionary Phase / Notes |
| :--- | :---: | :---: | :--- |
| **Baseline (Native GPT-2)** | **9.6361** | **0.0000** | **Default state without residual tuning** |
| Gen 01 | 9.5230 | +0.1130 | Exploration & step-size adaptation |
| Gen 02 | 9.3741 | +0.2619 | Covariance matrix expansion |
| Gen 03 | 9.0205 | +0.6155 | Substantial search space entry |
| Gen 05 | 8.4983 | +1.1378 | Exceeding 1.1 Loss reduction |
| **Gen 06** | **7.4059** | **+2.2301** | **Quantum jump optimization** |
| **Gen 10 - 15** | **7.3976** | **+2.2384** | **Global convergence & optimal vector locked** |

* **Total Search Time**: `112.79 seconds` on standard GPU/CPU environment.

---

### 2. Layer-Wise Fusion Topology ("Dumbbell Distribution")

```text
Layer 01  [6.5%]  ■■■
Layer 02  [11.7%] ■■■■■■  <-- Early Feature Core
Layer 03  [11.7%] ■■■■■■
Layer 04  [11.3%] ■■■■■
Layer 05  [4.9%]  ■■
Layer 06  [4.6%]  ■■
Layer 07  [3.6%]  ■■
Layer 08  [3.7%]  ■■
Layer 09  [11.7%] ■■■■■■  <-- Deep Aggregation
Layer 10  [7.6%]  ■■■
Layer 11  [4.7%]  ■■
Layer 12  [17.9%] ■■■■■■■■■ <-- Final Logits Core
