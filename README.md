

---

```markdown
# FedPAC-ME
### Federated Learning for Medical Image Analysis with Perspective-Aware Contrastive Learning and Mixture of Experts

Official implementation of **FedPAC-ME**, a federated learning framework that integrates **Multi-Perspective Contrastive Learning (MPCL)** with a **Mixture-of-Experts (MoE)** personalization module for robust, privacy-preserving medical image segmentation under heterogeneous (non-IID) client settings.

This repository accompanies our manuscript submitted to **Nature Scientific Reports**.

---

## 🔍 Key Features

- ✅ **Federated learning** with heterogeneous (non-IID) medical clients  
- ✅ **Multi-Perspective Contrastive Learning (MPCL)** for representation alignment  
- ✅ **Mixture-of-Experts (MoE)** for client-specific personalization  
- ✅ **Semi-supervised learning** with labeled and unlabeled data  
- ✅ Designed for **reproducibility** and **privacy-aware medical imaging research**

---

## 🧠 Framework Overview

FedPAC-ME jointly addresses:
1. **Client drift and data heterogeneity** via contrastive alignment across perspectives.
2. **Personalization–generalization trade-off** using a dynamic Mixture-of-Experts module.
3. **Limited annotations** through semi-supervised federated training.

All experiments reported in the manuscript can be reproduced using the scripts and configurations provided in this repository.

---

## 📁 Repository Structure

```

FedPAC-ME/
│
├── src/
│   ├── federated/        # Federated training and aggregation strategies
│   ├── models/           # Network architectures (MPCL + MoE)
│   ├── contrastive/      # Contrastive objectives and augmentations
│   ├── utils/            # Data loading, preprocessing, metrics, logging
│   ├── evaluation/       # Evaluation and statistical analysis
│   └── main.py           # Main experiment entry point
│
├── configs/
│   ├── fed_config.yaml   # Federated learning settings
│   ├── model_config.yaml # Model hyperparameters
│   └── data_config.yaml  # Dataset paths and augmentation policies
│
├── data/                 # Dataset placeholder (not included)
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation

### Environment Setup

- Python ≥ 3.8  
- PyTorch ≥ 1.12  

```bash
git clone https://github.com/yourusername/FedPAC-ME.git
cd FedPAC-ME

python -m venv fedpacme_env
source fedpacme_env/bin/activate        # Linux / macOS
fedpacme_env\Scripts\activate           # Windows

pip install -r requirements.txt
````

---

## 🗂️ Dataset Preparation

Datasets are **not included** due to licensing restrictions.

1. Download the dataset(s) used in the manuscript (e.g., BraTS).
2. Place them under the `data/` directory **or** specify custom paths in:

   ```
   configs/data_config.yaml
   ```
3. Configure:

   * Number of federated clients
   * Labeled / unlabeled data ratio
   * Data distribution strategy (IID / non-IID)

---

## 🚀 Running Experiments

### Federated Training

```bash
python src/main.py --config configs/fed_config.yaml
```

---

## 📊 Evaluation

Evaluate a trained model checkpoint:

```bash
python src/evaluation/eval.py --checkpoint path/to/model.pth
```

---

## 🔁 Reproducibility

All experiments are fully reproducible.

To fix randomness:

```bash
python src/main.py --config configs/fed_config.yaml --seed 42
```

Random seeds, optimizer settings, and training schedules are documented in the configuration files.

---

## 📌 Citation

If you use this repository in your research, please cite:

```
Federated Learning Framework for Medical Image Analysis with Perspective-Aware Contrastive and Mixture of Experts
K. Hemalatha, Shradhanjali Das
Scientific Reports (Nature)
DOI: To be announced
```

---

## 📜 License

This repository is intended for **academic and research use only**.

Please see the `LICENSE` file for details.

```

---

### Why this version works
- ✅ Clear **hierarchy**
- ✅ Quick **feature scan**
- ✅ Professional tone (matches *Scientific Reports*)
- ✅ GitHub-friendly formatting
- ✅ No unnecessary verbosity

If you want, next I can:
- add **badges** (PyTorch, Python, License)
- add a **framework diagram section**
- tailor it for **open-source visibility**
- create a **“Getting Started (5 minutes)”** section

Just tell me what goal you want this repo to serve.
```
