# MNIST Flask Application

A Flask application demonstrating MNIST digit recognition using PyTorch.

![Model Benchmarks](https://github.com/anuwrag/101_projects/workflows/Model%20Benchmarks/badge.svg)
![Build Status](https://github.com/anuwrag/101_projects/workflows/Build/badge.svg)

## Features

- Two-layer CNN architecture with less than 25,000 parameters
- Real-time training visualization
- Bootstrap-based responsive UI
- Live training metrics display

## Model Architecture

- Input: 28x28 grayscale images
- Conv1: 8 filters, 3x3 kernel
- MaxPool1: 2x2
- Conv2: 16 filters, 3x3 kernel
- MaxPool2: 2x2
- FC: 400 → 10

Total Parameters: 5,258

## Benchmarks

- Parameter count: < 25,000
- Accuracy: > 95% in 1 epoch

## Setup

1. Clone the repository:

