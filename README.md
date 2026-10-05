# Braintumor-MRI-uncertainty

# Uncertainty-Aware Brain Tumor MRI Classification

A deep learning project to classify brain MRI scans into four classes: glioma, meningioma, pituitary tumor, and no tumor.

## Project Goal
Build a dual-branch attention CNN (two pretrained backbones with CBAM) that:
- Estimates prediction uncertainty using Monte Carlo Dropout
- Calibrates confidence scores using temperature scaling
- Validates its decisions using two explainability methods: SHAP and Grad-CAM
- Is compared against single-backbone baselines with an ablation study

## Dataset
Brain Tumor MRI Dataset by Masoud Nickparvar (7,023 images, 4 classes):
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Status
Week 1: GitHub setup and dataset analysis.
