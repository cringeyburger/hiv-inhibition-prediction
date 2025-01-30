# MoleculeNet-HIV Prediction with Fine-Tuned BERT and LoRA <!-- omit in toc -->

## Overview <!-- omit in toc -->

This project focuses on predicting the HIV inhibition potential of drug-like molecules using SMILES (Simplified Molecular Input Line Entry System) strings. By fine-tuning a pre-trained BERT model with LoRA (Low-Rank Adaptation), I plan to achieve great performance with high accuracy and efficiency. The project also incorporates custom preprocessing pipelines for data preparation and a comparison with traditional machine learning models.

## Table of Content <!-- omit in toc -->

- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data](#data)
  - [Model](#model)
- [Results](#results)
- [Useage](#useage)
- [Future Work](#future-work)

### Installation

To be written.

### Dataset

The project uses the [MoleculeNet-HIV](https://moleculenet.org/datasets-1) dataset, which contains molecular SMILES strings labeled for their HIV inhibition potential.

As recommended, the dataset is broken into Bemis-Murcko scaffolds, and then split into train-test datasets.

It is to be noted that one can use [Deepchem library](https://deepchem.readthedocs.io/en/latest/get_started/installation.html) to quickly use any supported dataset, but I found that the HIV dataset was not operational. Hence, I made my own scaffolding and train-split functions, based on the [Deepchem](https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py) and [RDKit](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Scaffolds/MurckoScaffold.py) implementations.

### Methodology

#### Data

Scaffold splitting is essential and is preferred over random splitting in certain scenarios (like drug discovery) because it provides a more realistic and rigorous evaluation of a model's ability to generalize to unseen chemical structures.

Random splitting might give better results in terms of classification, but due to the realistic nature of scaffold splitting, I chose that.

After the data is split into train-test datasets, I work on either converting the SMILES to [Circular (Morgan) fingerprints](https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/circular_fingerprint.py), or tokenize it using the BERT tokenizer.

#### Model

Our goal is to check the reliability of transformers in drug discovery. Hence, I compare the performance of BERT with two other ML models- random forest and XGBoost.

### Results

To be written

### Useage

To be written

### Future Work

The project is just in its initial stages, and a lot of work needs to be done:

- [ ] Implement better methods:
  - [ ] Finetuning the BERT model
  - [ ] Make the models more modularized
  - [ ] Look into hyperparameter tuning
- [ ] Use better tracking methods to display results
- [ ] Compare with other popular models like ANNs, LSTM, LGBM, etc.
- [ ] Create an ensemble to output the best results possible

These are preliminary tasks that will be completed with time. Reading literature and being upto date is the bare minimum.
