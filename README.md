# 🎵 Music Genre Classification

This project focuses on classifying music genres using the GTZAN dataset by experimenting with different neural network architectures. The dataset includes audio tracks from 10 genres and their corresponding MEL spectrograms.

The study explores a progression of models: starting from a simple fully connected network, moving to convolutional networks, then to LSTM models, and finally employing a GAN for data augmentation. The data was split into training (70%), validation (20%), and testing (10%) sets, and performance was evaluated across multiple experiments.

---

## 📁 Dataset

- **Name:** [GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 
- **Genres:** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock  
- **Format:** 30-second audio clips (.wav) and MEL spectrograms  
- **Preprocessing:**
  - Silence removal
  - 1-second segmentation
  - MFCC feature extraction
- **Split:**
  - Training: 70%
  - Validation: 20%
  - Testing: 10%

---

## 🧠 Models

### 🔹 Net1: Fully Connected Network
- Two hidden layers: 128 and 64 neurons
- ReLU activation
- Input: Flattened spectrograms (shape: 3×180×180)
- Optimizer: Adam (lr=0.0001)
- Loss: Cross-entropy

### 🔹 Net2: Basic CNN
- 4 convolutional layers (filters: 32 → 64 → 128 → 256)
- Max pooling after 2nd and 4th layers
- Fully connected layers: 256 → 10
- ReLU activation
- Optimizer: Adam (lr=0.0001)

### 🔹 Net3: CNN + BatchNorm
- Same architecture as Net2, but with batch normalization after each convolution
- Helped stabilize training and reduce overfitting

### 🔹 Net4: CNN + BatchNorm + RMSProp
- Same as Net3 but uses RMSProp instead of Adam
- Slight decrease in performance

### 🔹 Net5: LSTM with MFCC
- Audio preprocessed to remove silence and split into 1s segments
- MFCCs used as input
- Architecture:
  - 2 LSTM layers (512, 256)
  - Fully connected layers: 128 → 64 → 10
- Optimizer: Adam (lr=0.0001)
- Early stopping applied

### 🔹 Net6: LSTM + GAN-based Data Augmentation
- GAN based on architecture from paper [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208)
- Generator: Dense projection + 5 ConvTranspose1D layers (output shape: waveform)
- Discriminator: 5 Conv1D layers + Dense output
- 27,980 generated samples (2,798 per genre)
- Classifier identical to Net5 but trained on original + generated data

---

## 📊 Results

### 📈 Spectrogram-Based Models

| Model | Accuracy (50 epochs) | Accuracy (100 epochs) |
|-------|----------------------|------------------------|
| Net1  | 0.4257               | 0.4950                 |
| Net2  | 0.6039               | 0.5346                 |
| Net3  | 0.6138               | 0.6138                 |
| Net4  | 0.5445               | 0.5346                 |

### 🔊 Audio-Based Models (MFCC)

| Model | Accuracy | Early Stopping Epoch |
|-------|----------|----------------------|
| Net5  | 0.7954   | 27                   |
| Net6  | 0.8291   | 18                   |

- Raw audio input (without MFCC): Accuracy < 0.15
- Using MFCC significantly improved performance

---

## 📉 Observations

- CNNs outperform fully connected networks on spectrograms.
- Batch normalization (Net3) helped maintain consistent accuracy over more epochs.
- RMSProp (Net4) did not improve performance compared to Adam.
- LSTM with MFCC (Net5) provided strong results for time-series audio data.
- GAN-based augmentation (Net6) further boosted accuracy and reduced training time.

---
