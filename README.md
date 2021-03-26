# Genrative Adversarial Networks for De Novo Molecular Design
This repository contains the original PyTorch implementation of the paper 'Generative Adversarial Networks for De Novo Molecular Design'.

## A. Requirements
Anaconda (>= 4.8.3)\
PyTorch (tested on 1.7.0)\
PyTorch Lightning (tested on 1.1.2)\
Tensorboard (tested on 2.4.0)

## B. Dataset
Donwlonad from the following link: https://www.ebi.ac.uk/chembl/ \
Or change the ChEMBL version and data save format to csv from the following link https://github.com/BenevolentAI/guacamol/blob/master/guacamol/data/get_data.py and run.\
Use make_data.ipynb to create the data to be used for SMILES-MaskGAN and save it as a pickle.

## C. Reference Code
Refer to the https://github.com/jerinphilip/MaskGAN.pytorch link for the PyTorch-based MaskGAN source code. 
Fairseq: This code uses the package provided at the https://github.com/pytorch/fairseq link.

## D. SMILES-MaskGAN training
Command line: python run_maskgan.py
