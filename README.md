# 3D GAN for MRI B1+ Field Prediction

## Overview

This project implements a simplified 3D Generative Adversarial Network (GAN) 
for predicting B1+ transmit field maps from standard MRI inputs.

The goal is to demonstrate how deep learning can estimate quantitative MRI parameters 
without direct field mapping acquisition.

---

## Problem

B1+ inhomogeneity affects quantitative MRI measurements. 
Direct B1+ mapping can be time-consuming or unavailable in clinical settings.

This project explores whether a 3D GAN can learn to predict B1+ maps 
from structural MRI contrasts.

---

## Model Architecture

- 3D U-Net style Generator
- 3D Patch-based Discriminator
- Adversarial Loss
- L1 Reconstruction Loss

---

## Tech Stack

- Python
- PyTorch
- NumPy
- NIfTI MRI handling

---

## Status

Prototype / Demonstration version

---

## Future Work

- Add training pipeline example
- Add evaluation metrics (SSIM, MAE)
- Add inference script
