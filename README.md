# Federated Learning Framework for Medical Image Analysis with Perspective-Aware Contrastive and Mixture of Experts (FedPAC-ME)
This repository contains the official implementation of FedPAC-ME, a federated learning framework that integrates Multi-Perspective Contrastive Learning (MPCL) with a Mixture-of-Experts (ME) personalization module to achieve robust, privacy-preserving medical image segmentation across heterogeneous clients.

The code supports federated training with labeled and unlabeled data, multi-perspective augmentations, contrastive alignment, and client-specific expert routing. It is built for reproducibility and fully aligned with the code policies of Nature Scientific Reports.

## 🧠 Overview
This repository contains the official implementation of FedPAC-ME, a federated learning framework designed for multi-modal medical image analysis. The project includes data preprocessing, visualization, model development, and federated training for 3D MRI volumes such as those in the BraTS2020 dataset.

FedPAC-ME integrates:

    Perspective-Aware Contrastive Learning (PAC-L)
    Mixture of Experts (ME) for personalization
    Federated Aggregation (FedAvg-based)
    Multi-modal image processing for T1, T1ce, T2, and FLAIR

This repository supports all the preprocessing and visualization techniques demonstrated in the Google Colab.

## 🚀 Features

Automatic download and extraction of multi-modal MRI datasets

Advanced visualization:

1. 2D slices
2. Multi-modality grids
3. Segmentation mask overlays
4. 3D volumetric rendering
   
Preprocessing pipeline:

1. Normalization
2. Smoothing
3. Resampling
4. Slice-wise extraction

Federated Learning components (simulation-ready)

All experiments presented in the manuscript can be reproduced using the scripts and configuration files provided here.

# 📁 Repository Structure
````
📦 FedPAC-ME/
│
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   ├── preprocess/
│   │   ├── dataloaders.py
│   │   ├── dataset.py
│   │   └── split_clients.py
│   ├── visualization/
│   │   ├── visualize_slices.py
│   │   ├── plot_modalities.py
│   │   ├── histogram_plots.py
│   │   └── segmentation_plots.py
│   ├── models/
│   │   ├── attention.py
│   │   ├── attention_contrastive_model.py
│   │   ├── loss.py
│   │   ├── moe.py
│   │   └── mpda.py
│   ├── training/
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── federated/
│       ├── fedavg.py
│       ├── client_simulator.py
│       └── aggregation.py
│
├── results/
│   ├── plots/
│   └── dataset/
│
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

````

## ⚙️ Installation

1. Clone the repository
````
git clone https://github.com/yourusername/FedPAC-ME.git
cd FedPAC-ME
````
3. Install dependencies
````
pip install -r requirements.txt
````
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

## 📥 Dataset Preparation
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





