# Eye Disease Prediction using DenseNet121

This project utilizes a convolutional neural network (CNN) model based on DenseNet121 to classify images into four eye disease categories. The project achieves high accuracy on a Kaggle dataset of eye disease images.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Results](#results)

## Overview

The Eye Disease Prediction project aims to classify images into the following categories:
- Cataract
- Diabetic Retinopathy
- Glaucoma
- Normal

Using DenseNet121 as the base model, the project achieves a validation accuracy of **92.88%** with a validation loss of **0.35**. The model is trained on 80% of the data and validated on the remaining 20%.

## Dataset

The dataset is sourced from Kaggle:
- [Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shardajadhav03/eye-disease-prediction.git
   cd eye-disease-prediction

2. **Install Required Libraries:**
   pip install -r requirements.txt
3. **Download the Dataset:** Use the Kaggle API to download the dataset:
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification

4. **Unzip the Dataset:**
   import zipfile
   with zipfile.ZipFile('eye-diseases-classification.zip', 'r') as zip_ref:
       zip_ref.extractall('eye-diseases-val')
## Project Structure
eye-disease-prediction/
├── data/               # Folder containing dataset images
├── models/             # Saved models
├── src/                # Source code for data loading, model, and evaluation
├── requirements.txt    # Required libraries
└── README.md           # Project documentation

## Model Architecture
The model is built on a pre-trained DenseNet121 model, which has been adapted to the four output classes for this dataset. The architecture includes:

- DenseNet121 base model: Pre-trained on ImageNet, with the top layer removed.
- Additional Layers:
  - Max Pooling
  - Batch Normalization
  - Dense layers with dropout for regularization

## Model Hyperparameters
- Input Shape: (224, 224, 3)
- Batch Size: 32
- Classes: 4

## Compiling the Model
The model is compiled with:

- Optimizer: RMSprop
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy

## Training the Model
1. **Data Preprocessing:** The dataset is split into training and validation sets (80% for training and 20% for validation) using ImageDataGenerator.

2. **Model Training:** The model is trained with the following callbacks:
   - Model Checkpoint: Saves the best model based on validation accuracy.
   - ReduceLROnPlateau: Reduces the learning rate if the validation loss plateaus.
3. **Validation Accuracy:** After training, the model achieves 92.88% accuracy on the validation set.

## Results
- **Validation Accuracy:** 92.88%
- **Validation Loss:** 0.35

