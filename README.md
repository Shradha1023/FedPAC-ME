# Federated Learning Framework for Medical Image Analysis with Perspective-Aware Contrastive and Mixture of Experts (FedPAC-ME)
This repository contains the official implementation of FedPAC-ME, a federated learning framework that integrates Multi-Perspective Contrastive Learning (MPCL) with a Mixture-of-Experts (ME) personalization module to achieve robust, privacy-preserving medical image segmentation across heterogeneous clients.

The code supports federated training with labeled and unlabeled data, multi-perspective augmentations, contrastive alignment, and client-specific expert routing. It is built for reproducibility and fully aligned with the code policies of Nature Scientific Reports.

# 🧠 Overview
This repository contains the source code used to implement the FedPAC-ME framework described in the associated manuscript. The framework integrates (i) multi-perspective contrastive learning for representation alignment and (ii) a mixture-of-experts module for client-specific personalization within a federated learning environment. The codebase supports semi-supervised training using both labeled and unlabeled medical imaging data and is designed to operate under heterogeneous (non-IID) client distributions.

All experiments presented in the manuscript can be reproduced using the scripts and configuration files provided here.

# 📁 Repository Structure
````
📦 FedPAC-ME/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE   
│
├── notebooks/
│   └── version1_8.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   ├── visualization/
│   │   ├── visualize_slices.py
│   │   ├── plot_modalities.py
│   │   ├── histogram_plots.py
│   │   └── segmentation_plots.py
│   ├── models/
│   │   ├── unet.py
│   │   └── other_models.py
│   ├── training/
│   │   ├── train.py
│   │   └── metrics.py
│   └── federated/
│       ├── split_clients.py
│       ├── fedavg.py
│       ├── utils.py
│       └── simulation.py
│
├── scripts/
│   ├── run_preprocessing.sh
│   ├── run_training.sh
│   └── run_visualization.sh
│
└── results/
    ├── sample_plots/
    ├── models/
    └── logs/

````

# ⚙️ Installation
## 🚀 Environment Setup

````

- Python ≥ 3.8  
- PyTorch ≥ 1.12  

```bash
git clone https://github.com/yourusername/FedPAC-ME.git
cd FedPAC-ME
python -m venv fedpacme_env
source fedpacme_env/bin/activate      # Linux/macOS  
fedpacme_env\Scripts\activate         # Windows
pip install -r requirements.txt

````

## 🗂️ Dataset Preparation
Datasets are not included in this repository. Users should:

Download the dataset(s) used in the manuscript
Place the files under the data/ directory or specify alternative paths in configs/data_config.yaml.
If federated partitioning is required, specify the number of clients, labeled/unlabeled ratios, and distribution settings in the YAML configuration.
Running Experiments
Federated Training
To reproduce the main experimental results:
```
python src/main.py --config configs/fed_config.yaml
```
# 📊 Evaluation Procedures
To evaluate a trained global model:
```
python src/evaluation/eval.py --checkpoint path/to/model.pth
```
# 🔁 Reproducibility
All random seeds used in the manuscript can be set via command-line arguments or within the configuration files. To reproduce the exact experimental conditions:
```
python src/main.py --config configs/fed_config.yaml --seed 42
```
# 📌 Citation
Please cite the associated manuscript when using this repository:

```
Federated Learning Framework for Medical Image Analysis with Perspective-Aware Contrastive and Mixture of Experts
K. Hemalatha, Shradhanjali Das
Scientific Reports (Nature)
DOI: To be announced
```

# 📜 License
This repository is intended for **academic and research use only**.

Please see the `LICENSE` file for details.





