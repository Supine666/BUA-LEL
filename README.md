# BUA-LEL

Official PyTorch implementation of:

**BUA-LEL: Boundary-Uncertainty-Aware Lesion Evidence Learning for Breast Ultrasound Segmentation and Molecular Subtyping**

BUA-LEL performs joint breast ultrasound lesion segmentation and molecular subtype prediction using multimodal inputs:

- Ultrasound images (X ∈ ℝ^{H×W×3})  
- Clinical variables (c ∈ ℝ^{d_c})  

---

## 1. Method Overview

BUA-LEL consists of the following modules:

1. **MedSAM encoder**: extracts multi-scale features for lesion segmentation and morphology-aware classification.
2. **Boundary-uncertainty graph**: refines coarse lesion priors and explicitly encodes uncertain lesion margins.
3. **Zonal pooling**: decomposes the refined lesion into core, boundary, and peritumoral regions, producing regional morphology tokens.
4. **Morphology-Clinical Heterogeneous Graph**: integrates lesion morphology tokens with clinical tokens for relation-aware reasoning.

**Optimization objective**:

\[
\mathcal{L}_{total} = \mathcal{L}_{seg} + \mathcal{L}_{cls} + \text{task-uncertainty weighting} + \lambda_{anchor} \mathcal{L}_{anchor}
\]

- Segmentation loss: BCE + Dice  
- Classification loss: cross-entropy with optional label smoothing  
- Anchor-constrained boundary regularization  
- Learnable task uncertainty weighting  

---

## 2. Repository Structure

```text
BUA-LEL-main/
|-- datasets/
|   |-- __init__.py
|   `-- bua_dataset.py
|-- engines/
|   |-- losses.py
|   `-- train_eval.py
|-- models/
|   |-- __init__.py
|   |-- bua_lel.py
|   |-- clinical_graph.py
|   |-- morph_clin_hetero_graph.py
|   |-- backbones/
|   |   |-- __init__.py
|   |   `-- medsam_encoder.py
|   |-- boundary/
|   |   |-- __init__.py
|   |   `-- reliability_anchor_bgr.py
|   |-- heads/
|   |   |-- __init__.py
|   |   |-- seg_decoder.py
|   |   `-- subtype_head.py
|   `-- roi/
|       |-- __init__.py
|       `-- multi_region_pooling.py
|-- scripts/
|   `-- run_cv.py
|-- segment_anything/
|-- utils/
|   |-- meters.py
|   |-- metrics_cls.py
|   |-- metrics_seg.py
|   |-- optim.py
|   |-- preprocess_fold.py
|   |-- roc.py
|   `-- seed.py
|-- work_dir/
|   `-- MedSAM/
|       `-- medsam_vit_b.pth
|-- requirements.txt
`-- README.md
```

---

## 3. Environment

**Tested stack**:

- Python 3.10  
- PyTorch 2.0.1 + CUDA 11.8  
- torchvision 0.15.2  
- torch-geometric 2.7.0  
- OpenCV 4.7.0  
- scikit-learn 1.2.1  
- pandas 1.5.3  

**Installation**:

```bash
conda create -n bua_lel python=3.10 -y
conda activate bua_lel

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 4. Data Preparation

BUA-LEL was evaluated on five datasets in the paper, including two private clinical breast ultrasound cohorts and three public benchmarks. The two in-house clinical cohorts from Peking Union Medical College Hospital and Beijing Longfu Hospital are not publicly released due to institutional privacy, ethical, and data-sharing restrictions. Researchers interested in academic use may contact the corresponding author for potential access, subject to approval by the data-owning institutions.

The three public datasets used in the paper can be downloaded from their official sources or through the corresponding references cited in the manuscript:

- **BrEaST**: public ultrasound–BI-RADS multimodal benchmark.
- **BUSI**: public breast ultrasound image dataset for lesion segmentation and classification.
- **ISIC 2018**: public skin lesion image–metadata benchmark used for cross-domain validation.

Please refer to the dataset citations in the paper for detailed download links, licenses, and usage conditions. In the manuscript, BUA-LEL is evaluated on two in-house cohorts and three public benchmarks, namely HER2USC, LMNUSC, BrEaST, BUSI, and ISIC 2018. :contentReference[oaicite:0]{index=0}

### Expected Data Format

For running this repository on your own dataset or on downloaded public datasets, organize the data as follows:

```text
data/
|-- images/
|   |-- 001.bmp
|   |-- 002.bmp
|   `-- ...
|-- masks/
|   |-- 001.png
|   |-- 002.png
|   `-- ...
`-- clinical.xlsx

---

## 5. Training & Evaluation

**Run 5-fold cross-validation**:

```bash
python scripts/run_cv.py
```

- Batch size: 4  
- Epochs: 50  
- Optimizer: AdamW, LR scheduler: CosineAnnealing  
- Mixed precision enabled  
- Early stopping patience: 12 epochs  

**Outputs**:

- Fold-specific best checkpoints (`checkpoints_bul/best_fold*.pth`)  
- Fold metrics CSV (`fold*/fold*_metrics.csv`)  
- Out-of-fold ROC plots (`OOF_ROC.png`)  
- Aggregated summary CSV (`paper_table_mean_std.csv`)  

---

## 6. Evaluation Metrics

| Task | Dice | mIoU | ACC | Macro-F1 | AUC |
|------|------|------|-----|----------|-----|
| Segmentation | 0.742 ± 0.042 | 0.677 ± 0.064 | - | - | - |
| Classification | - | - | 0.832 ± 0.048 | 0.820 ± 0.043 | 0.894 ± 0.058 |

Metrics reported as mean ± standard deviation across 5-fold patient-level cross-validation.

---

## 7. Reproducibility Checklist

1. Python 3.10, PyTorch 2.0.1  
2. MedSAM checkpoint at `work_dir/MedSAM/medsam_vit_b.pth`  
3. Dataset prepared as above with consistent patient IDs  
4. Fixed seed = 42  
5. Default TrainConfig used unless performing ablation  

---

## 8. Citation

```bibtex
@article{YeBUALEL2026,
  title   = {Boundary-Uncertainty-Aware Lesion Evidence Learning for Breast Ultrasound Segmentation and Molecular Subtyping},
  author  = {Ye, Jinlin and Hu, Deming and Ge, Zhongyu and Li, Ziqi and Yuan, Shouhang and Liu, Yuhan and Yang, Liang and Ren, Shangjie and Wang, Changjun and Zhou, Yidong and Zhang, Wei},
  year    = {2026},
  note    = {Under review}
}
```

# License

This project is released under the **MIT License**.
