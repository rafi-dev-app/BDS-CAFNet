 
<div align="center">

# BDS-CAFNet

### DenseNet–Swin Cross-Attention Fusion Network

**An Explainable Hybrid Deep Learning Framework for Multi-Class Dermatological Image Classification**

<p align="center">
  <img src="docs/assets/bds-cafnet-banner.png.jpg" width="100%" alt="BDS-CAFNet Project Banner">
</p>

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-CNN%20%2B%20Transformer-7c3aed)](#model-architecture)
[![Classes](https://img.shields.io/badge/Classes-15-0891b2)](#dataset)
[![Explainable AI](https://img.shields.io/badge/Explainable-AI-16a34a)](#explainable-ai)

**Combining local visual features, global contextual information, and cross-attention fusion for explainable dermatological image classification.**

</div>

---

## Overview

BDS-CAFNet is an explainable hybrid deep learning framework designed for multi-class dermatological image classification. The framework combines the local feature extraction capability of **DenseNet201** with the global contextual modelling capability of **Swin Transformer-Tiny** through a dedicated **Cross-Attention Fusion Module (CAFM)**.

The proposed architecture is developed to address the visual similarities among dermatological conditions and the challenges of limited primary clinical image availability in resource-constrained healthcare environments such as Bangladesh.

Unlike a conventional single-backbone model, BDS-CAFNet uses two parallel feature extraction branches and a structured attention-based fusion strategy to learn complementary representations from dermatological images.

## Key Highlights

- **Hybrid Architecture:** Parallel DenseNet201 and Swin Transformer-Tiny branches.
- **Cross-Attention Fusion:** Integrates local and global feature representations using CAFM.
- **Spatial Feature Refinement:** Coordinate Attention enhances informative spatial representations.
- **Explainable AI:** Grad-CAM, Grad-CAM++, CAFM attention visualization, and t-SNE analysis.
- **Primary Clinical Dataset:** 1,850 original images collected from 210 consenting patients.
- **Multi-Class Classification:** 15 dermatological categories.
- **Class-Balanced Training:** 7,500 augmented images across training, validation, and test partitions.

## Research Performance

<div align="center">

| Evaluation Metric | Performance |
|:---|---:|
| Test Accuracy | **97.87%** |
| Macro Precision | **97.97%** |
| Macro Recall | **97.87%** |
| Macro F1-Score | **97.86%** |
| Macro AUC | **0.9998** |
| Five-Fold CV Accuracy | **99.04 ± 0.39%** |
| Five-Fold CV Macro AUC | **0.9998** |

</div>

The held-out test results and five-fold cross-validation results are reported separately. The cross-validation results are based on the study's augmented dataset.

---

## Model Architecture

BDS-CAFNet follows a dual-branch architecture in which two complementary backbones independently extract dermatological image features. Their representations are integrated through cross-attention and further refined before final classification.

<div align="center">

<img src="docs/architecture.svg" width="100%" alt="BDS-CAFNet Dual-Branch Architecture">

**Figure 1. Overall architecture of the proposed BDS-CAFNet framework.**

</div>

### 1. Input Image

The input dermatological image is preprocessed and resized to **224 × 224 × 3** before being forwarded to both feature extraction branches.

### 2. DenseNet201 Branch — Local Feature Extraction

DenseNet201 is employed as the convolutional feature extraction branch.

It captures important local visual patterns, including:

- Skin texture and colour variations.
- Lesion boundaries and edges.
- Fine-grained visual structures.
- Local dermatological characteristics.

### 3. Swin Transformer-Tiny Branch — Global Contextual Modelling

Swin Transformer-Tiny is used to learn global contextual and spatial relationships from the input image.

Its shifted-window self-attention mechanism helps capture:

- Long-range contextual information.
- Spatial relationships between image regions.
- Global lesion distribution.
- Broader visual patterns.

### 4. Cross-Attention Fusion Module (CAFM)

The Cross-Attention Fusion Module is the central fusion component of BDS-CAFNet.

It integrates the complementary feature representations extracted by DenseNet201 and Swin Transformer-Tiny.

The fusion process uses:

- **Query (Q):** DenseNet201 local features.
- **Key (K):** Swin Transformer global features.
- **Value (V):** Swin Transformer global features.

Multi-head cross-attention allows the local feature representation to interact with global contextual information, producing a more integrated feature representation.

### 5. Coordinate Attention

Following cross-attention fusion, Coordinate Attention is applied to refine the fused representation.

It helps emphasize informative spatial regions and preserve relevant positional information for subsequent classification.

### 6. Classification Head

The refined feature representation is passed through a fully connected classification layer followed by Softmax activation.

The final output represents the predicted probabilities across **15 dermatological classes**.

---

## Dataset

The proposed framework was evaluated using a primary clinical dermatological dataset collected at the **Bangladesh Institute of Dermatology, STD and AIDS**.

| Dataset Information | Description |
|:---|:---|
| Original Images | 1,850 |
| Patients | 210 |
| Target Categories | 15 |
| Training Split | 80% |
| Validation Split | 10% |
| Testing Split | 10% |
| Final Training Images | 6,000 |
| Final Validation Images | 750 |
| Final Testing Images | 750 |
| Total Augmented Images | **7,500** |

The original images were stratified into training, validation, and testing partitions before augmentation. Augmentation was then performed independently within each partition to prevent augmented-image overlap across the partitions.

### Dataset Samples

<div align="center">

<img src="docs/results/dataset_samples.png" width="100%" alt="Dermatological Dataset Samples">

**Figure 2. Representative samples from the dermatological dataset.**

</div>

### Data Preprocessing

The preprocessing pipeline was designed to improve image quality and enhance relevant dermatological features.

The applied operations include:

1. Contrast Limited Adaptive Histogram Equalization (CLAHE).
2. Gamma Correction.
3. Non-Local Means Denoising.
4. Sobel Edge Enhancement.
5. Image Resizing to 224 × 224.
6. ImageNet Normalization.

<div align="center">

<img src="docs/results/preprocessing_grid.png" width="100%" alt="Image Preprocessing Visualization">

**Figure 3. Visual representation of the image preprocessing pipeline.**

</div>

### Data Augmentation

The augmentation strategy was applied independently to each dataset partition to improve class balance and support model generalization.

| Partition | Before Augmentation | After Augmentation |
|:---|---:|---:|
| Training | 1,476 | 6,000 |
| Validation | 180 | 750 |
| Testing | 194 | 750 |
| **Total** | **1,850** | **7,500** |

The augmentation operations included horizontal flipping, bounded rotation, translation, and brightness adjustment.

---

## Experimental Results

The proposed BDS-CAFNet framework was evaluated using a held-out test set and five-fold cross-validation.

### Confusion Matrix

The confusion matrix presents the class-wise prediction distribution and illustrates the classification performance across the 15 dermatological categories.

<div align="center">

<img src="docs/results/confusion_matrix.png" width="90%" alt="BDS-CAFNet Confusion Matrix">

**Figure 4. Confusion matrix of BDS-CAFNet on the held-out test set.**

</div>

### ROC Curve Analysis

The Receiver Operating Characteristic (ROC) curve illustrates the discrimination capability of the proposed framework across multiple dermatological classes.

<div align="center">

<img src="docs/results/roc_curve.png" width="90%" alt="Multi-class ROC Curve">

**Figure 5. Multi-class ROC curve with macro AUC of 0.9998.**

</div>

### Training and Validation Curves

The training curves illustrate the model's optimization behaviour and its performance progression during the training process.

<div align="center">

<img src="docs/results/training_curve.png" width="90%" alt="BDS-CAFNet Training Curves">

**Figure 6. Training and validation performance curves.**

</div>

### Comparison with Individual Backbones

BDS-CAFNet was also compared with its individual backbone architectures to evaluate the contribution of hybrid feature learning and attention-based fusion.

| Model | Test Accuracy |
|:---|---:|
| DenseNet201 | 95.73% |
| Swin Transformer-Tiny | 94.27% |
| **BDS-CAFNet** | **97.87%** |

<div align="center">

<img src="docs/results/hybrid_comparison.png" width="90%" alt="Comparison with Individual Backbones">

**Figure 7. Performance comparison between BDS-CAFNet and individual backbone architectures.**

</div>

---

## Explainable AI

Interpretability is an important component of the proposed framework. Several visualization techniques were used to investigate the learned representations and identify influential image regions.

### Grad-CAM Visualization

Grad-CAM highlights image regions that contribute to the model's prediction by generating class-specific activation maps.

<div align="center">

<img src="docs/results/gradcam.png" width="100%" alt="Grad-CAM Visualization">

**Figure 8. Grad-CAM visualization of dermatological image predictions.**

</div>

### Grad-CAM++ Visualization

Grad-CAM++ provides an additional gradient-based visualization approach for examining important image regions associated with model predictions.

<div align="center">

<img src="docs/results/gradcam_plus_plus.png" width="100%" alt="Grad-CAM++ Visualization">

**Figure 9. Grad-CAM++ visualization of the proposed framework.**

</div>

### CAFM Attention Visualization

The CAFM attention maps provide insight into the interaction between local DenseNet201 features and global Swin Transformer features.

<div align="center">

<img src="docs/results/cafm_attention.png" width="100%" alt="CAFM Cross-Attention Visualization">

**Figure 10. Cross-attention visualization generated from the CAFM module.**

</div>

### t-SNE Feature Visualization

t-SNE is used to visualize the distribution of learned feature representations in a lower-dimensional space.

<div align="center">

<img src="docs/results/tsne_features.png" width="85%" alt="t-SNE Feature Visualization">

**Figure 11. t-SNE visualization of learned feature representations.**

</div>

---

## Model Complexity

The proposed architecture was designed and evaluated with consideration of computational resource limitations.

| Model Specification | Value |
|:---|---:|
| Total Parameters | 49,416,491 |
| Trainable Parameters | 46,046,889 |
| Frozen Parameters | 3,369,602 |
| Computational Complexity | 8.87 GMACs |
| Approximate FLOPs | 17.75 GFLOPs |
| Input Resolution | 224 × 224 × 3 |

## Training Configuration

The model was trained using a two-phase optimization strategy.

| Configuration | Value |
|:---|:---|
| Deep Learning Framework | PyTorch 2.10.0 |
| Python Version | 3.11 |
| Accelerator | NVIDIA Tesla T4 (16 GB) |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Batch Size | 16 |
| Random Seed | 42 |
| Training Strategy | Two-Phase Training |
| Phase 1 Learning Rate | 1 × 10⁻⁴ |
| Phase 2 Backbone Learning Rate | 1 × 10⁻⁵ |
| Phase 2 New Modules Learning Rate | 1 × 10⁻⁴ |
| Learning Rate Scheduler | Cosine Annealing |
| Minimum Learning Rate | 1 × 10⁻⁶ |
| Loss Function | 0.6 Focal Loss + 0.4 Dice Loss |
| Mixed Precision | Automatic Mixed Precision |
| Early Stopping Patience | 8 |

---

## Repository Structure

```text
BDS-CAFNet/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── docs/
│   ├── architecture.svg
│   │
│   ├── assets/
│   │   └── bds-cafnet-banner.png
│   │
│   └── results/
│       ├── dataset_samples.png
│       ├── preprocessing_grid.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── training_curve.png
│       ├── hybrid_comparison.png
│       ├── gradcam.png
│       ├── gradcam_plus_plus.png
│       ├── cafm_attention.png
│       └── tsne_features.png
│
├── src/
│   ├── models/
│   ├── modules/
│   ├── preprocessing/
│   └── evaluation/
│
├── notebooks/
│
└── requirements.txt
```

*The structure above represents a suggested organization for the repository. Actual folders and scripts should reflect the files included in the published implementation.*

## Reproducibility

The experimental configuration includes a fixed random seed, controlled training settings, and a two-phase optimization strategy.

For reproducibility, the implementation should retain the documented preprocessing operations, data partitioning strategy, augmentation settings, model configuration, and evaluation protocol.

## Limitations

Although BDS-CAFNet achieved strong results on the evaluated dataset, several limitations should be considered:

- The dataset contains a relatively limited number of original clinical images.
- The evaluation was conducted using data from a specific clinical collection.
- Performance on images from other hospitals and populations requires further external validation.
- Attention and gradient-based visualizations provide interpretability insights but do not independently establish clinical correctness.

Further evaluation using larger, multi-centre clinical datasets is necessary before considering real-world clinical deployment.

## Ethical Considerations

This research uses primary clinical dermatological images obtained through an ethical data collection process involving consenting patients.

Original patient images and identifiable clinical information are not included in this public repository. Any redistribution of clinical data requires appropriate patient consent and institutional authorization.

**BDS-CAFNet is a research framework and is not intended to replace professional dermatological diagnosis or clinical decision-making.**

## Citation

If you use or refer to this research, please cite the corresponding thesis or publication after its official bibliographic details are finalized.

```bibtex
@misc{bds_cafnet,
  title        = {BDS-CAFNet: DenseNet-Swin Cross-Attention
                  Fusion Network for Dermatological Image Classification},
  author       = {Mehedi Hasan Rafi and Md. Maruf Kadir},
  year         = {2026},
  note         = {Research project}
}
```

*Please update this citation if an official publication with a DOI or final bibliographic information becomes available.*

## Acknowledgements

The authors acknowledge the support of the Department of Computer Science and Engineering, Daffodil International University, and the clinical collaboration that made the primary dermatological image collection possible.

## Authors

**Mehedi Hasan Rafi**  
**Md. Maruf Kadir**

Department of Computer Science and Engineering  
Daffodil International University, Bangladesh

---

<div align="center">

### BDS-CAFNet

**Local Features. Global Context. Attention-Based Fusion.**

*Towards explainable and reliable dermatological image classification.*

</div>
