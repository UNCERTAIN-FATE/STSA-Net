# STSA-Net
This repository contains the official implementation of the paper:
**STSA-Net: Small Target Sensitive and Adaptive Sparse Network for Infrared Reconstruction**.

> **Abstract:** In infrared imaging, spatially proximate small targets undergo severe aliasing... This paper proposes STSA-Net, a structure-enhanced deep unfolding network... incorporating Small Target Sensitive Convolution (STSC) and Adaptive Sparse Threshold (AST) modules.

![Network Architecture](figures/architecture.png)
*(Note: You can upload your revised Figure 1 here)*

## 🚀 Highlights
- **Structure-Enhanced:** Introduces STSC to preserve small target details.
- **Adaptive Thresholding:** Uses AST for robust noise suppression.
- **SOTA Performance:** Outperforms DISTA-Net and other baselines on the CSIST-100K dataset.

## 🛠️ Requirements

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0

Install dependencies:
```bash
pip install -r requirements.txt
