# FSDL 2022 — Lab 01 (Modernized): Deep Neural Networks in PyTorch

A modernized, Colab-runnable rewrite of the FSDL 2022 Lab 01 notebook, covering
the fundamentals of building, training, and evaluating a neural network in
PyTorch using the FashionMNIST dataset.

## Overview

This notebook walks through the full PyTorch workflow end to end:

- Loading and exploring the FashionMNIST dataset
- Core tensor operations
- Building a simple Multi-Layer Perceptron (MLP) classifier
- Forward propagation, loss computation, and autograd/backpropagation
- A full training and validation loop with loss/accuracy tracking
- Plotting training history (loss and accuracy curves)
- Saving and loading model weights
- Running inference on single images, batches, and random samples

## Why "modernized"

The original FSDL 2022 lab notebooks rely on package versions and APIs that
have since changed. This version has been updated to run cleanly on current
package releases (`torch`, `torchvision`, `matplotlib`, `tqdm`) with no
legacy breakages, so it can be opened and run directly in Google Colab.

## Model

A simple feed-forward MLP (`SimpleMLP`):
- Flattens the 28x28 input image
- One or more hidden `Linear` layers
- Trained with `CrossEntropyLoss` and the `Adam` optimizer

## Requirements

```bash
pip install torch torchvision matplotlib tqdm
```

## Usage

Open `FSDL_22_Self_Lab01.ipynb` in Jupyter or Google Colab and run all cells
top to bottom. The dataset downloads automatically via `torchvision.datasets`.

## Results

After 5 epochs of training, the notebook reports training/validation loss
and accuracy, plots the training curves, and runs sample predictions on
held-out test images.

## Acknowledgments

Based on Lab 01 from the [Full Stack Deep Learning (FSDL) 2022](https://fullstackdeeplearning.com/) course.
