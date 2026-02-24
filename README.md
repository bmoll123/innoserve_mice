# MICE

**Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection**

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Environments](#environments)
* [📊 Data Preparation](#data-preparation)
* [🚀 Run Experiments](#run-experiments)
* [🔗 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)
* [📜 License](#license)

## Introduction
This repository provides PyTorch-based source code for PBAS,
a framework that enhances unsupervised anomaly detection by directionally synthesizing significant anomalies
without predefined texture properties, guided by a progressive decision boundary.
Here, we present a brief summary of PBAS's performance across several benchmark datasets.

|  PBAS   | MVTec AD | VisA  | MPDD  |
|:-------:|:--------:|:-----:|:-----:|
| I-AUROC |     |  |  |
| P-AUROC |     |  |  |

## Environments
Create a new conda environment and install required packages.
```
conda create -n mice_env python=3.11.5
conda activate mice_env
pip install -r requirements.txt
```
Experiments were conducted on NVIDIA GeForce RTX 3090 (24GB).
Same GPU and package version are recommended. 

## Data Preparation
The public datasets employed in the paper are listed below.
These dataset folders/files follow its original structure.

- MVTec AD ([Download link](https://www.mvtec.com/company/research/datasets/mvtec-ad/))
- VisA ([Download link](https://github.com/amazon-science/spot-diff/))
- MPDD ([Download link](https://github.com/stepanje/MPDD/))

## Run Experiments
For example, edit `./shell/run-mvtec.sh` to configure arguments `--datapath`, `--classes`, and hyperparameter settings.
Please modify argument `--test` to 'ckpt' / 'test' to toggle between training and test modes.

```
bash run-mvtec.sh
```
