# Flower Species Classification Project

## Overview

This project aims to classify flower species using pretrained deep learning models (EfficientNet, ResNet, and VGG). The classifier can identify the species of a flower from an image and provides a detailed classification.

## Features

- Image classification using pretrained models: EfficientNet, ResNet50, and VGG16.
- Supports image input in local file format.
- Implements early stopping to prevent overfitting.

## Prerequisites

- Python 3.8
- Anaconda or Miniconda for environment management

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/flower-species-classification.git
    cd flower-species-classification
    ```

2. **Create and activate a conda environment:**

    ```bash
    conda create --name flower-classifier python=3.8
    conda activate flower-classifier
    ```

3. **Install the required packages:**

    ```bash
    conda install pytorch torchvision numpy tqdm matplotlib
    ```

## Usage

1. **Prepare the data:**

   Ensure your data is organized in the following structure:
   
    ```
    data/
        train/
        valid/
        test/
    ```

2. **Train the classifier:**

    ```bash
    python train.py data_dir --save_dir checkpoints_folder --arch efficientnet --hidden_units 512 --learning_rate 0.001 --epochs 20 --gpu
    ```

3. **Evaluate the classifier:**

    Once the training is complete, the model will be saved as a checkpoint in the specified directory.

    ```bash
    python train.py data --topk 5 --category_names cat_to_name.json --gpu
    ```


## Acknowledgements

This project was created as part of the AWS AI ML Nanodegree program.