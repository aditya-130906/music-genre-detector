# Music Genre Classification using PyTorch

This project implements a deep learning model to classify music into genres using audio signals. The model is trained on the **GTZAN dataset** and leverages **Residual Networks (ResNet-style CNNs)** on Mel spectrograms for improved performance and stability.

---

##  Model Architecture

The model is built using a hybrid CNN + ResNet approach:

* Initial Convolution Layer (Conv2D + BatchNorm + ReLU)
* **Residual Blocks (ResNet-style skip connections)**

  * Residual Block 1: 16 → 32 channels
  * Residual Block 2: 32 → 64 channels
* Max Pooling after each block
* Adaptive Average Pooling (fixed output size)
* Fully Connected Layers
* Dropout for regularization

###  Key Feature: Residual Connections

Residual connections allow the model to:

* Learn deeper representations
* Avoid vanishing gradient problems
* Improve training stability and convergence

---

##  Audio Processing

* Audio resampled to **22,050 Hz**
* Converted to **mono**
* Fixed duration: **3 seconds per sample**
* Converted to **Mel Spectrograms**
* Log scaling using **AmplitudeToDB**
* Optional data augmentation:

  * Time shifting
  * Noise injection
  * Gain variation
  * SpecAugment (time & frequency masking)

---

##  Dataset

This project uses the **GTZAN Genre Classification Dataset**:

* 1000 audio files
* 10 genres (blues, jazz, rock, etc.)
* Each clip is 30 seconds long

---

##  Training Details

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate Scheduler: ReduceLROnPlateau
* Training/Validation/Test Split: 80% / 10% / 10%

---

##  Results

* Baseline CNN Accuracy: ~55%
* ResNet-based Model Accuracy: **~66–70%**

Residual connections improved:

* Convergence speed
* Validation stability
* Overall performance

---

##  Future Improvements

* Larger dataset or data augmentation strategies
* Attention mechanisms for audio features
* Pretrained audio models (e.g., transfer learning)
* Hyperparameter tuning

---

##  Tech Stack

* Python
* PyTorch
* Torchaudio

---

##  Summary

This project demonstrates how **ResNet-style architectures** can significantly improve performance in audio classification tasks by enabling deeper and more stable neural networks.

---
