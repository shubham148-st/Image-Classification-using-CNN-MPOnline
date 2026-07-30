# Image Classification using CNN

# AUTHOR 
NAME: SHUBHAM PURBEY 
REG.NO: 23MIM10040

## Objective

Develop a Convolutional Neural Network (CNN) model for an animal welfare organization to automatically classify pet images into **Cats** and **Dogs** categories. The goal is to achieve high classification accuracy using deep learning techniques on image data.

## Dataset

**Cats vs Dogs Dataset** from Kaggle:

[Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

> **Note:** The dataset is not included in this repository. Please download it from the Kaggle link above and extract it into a `dataset/` folder in the project root directory.

### Expected folder structure after download:

```
Assignment-9/
├── Assignment-9.py
├── README.md
└── dataset/
    ├── cats/
    │   ├── cat_0001.jpg
    │   ├── cat_0002.jpg
    │   └── ...
    └── dogs/
        ├── dog_0001.jpg
        ├── dog_0002.jpg
        └── ...
```

##  Libraries Used

| Library | Purpose |
|---|---|
| **TensorFlow / Keras** | Building, training, and evaluating the CNN model |
| **NumPy** | Numerical computations and array operations |
| **Matplotlib** | Plotting sample images, training curves, and graphs |
| **Seaborn** | Visualizing the confusion matrix heatmap |
| **scikit-learn** | Computing precision, recall, F1-score, and confusion matrix |

### Installation

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Methodology

### 1. Data Understanding
- Explored the dataset folder structure and class distribution.
- Displayed sample images with their class labels.
- Identified the number of classes (2: Cats and Dogs), image dimensions, and total image count.

### 2. Data Preprocessing
- **Resized** all images to **128 × 128** pixels for uniform input dimensions.
- **Normalized** pixel values to the range **[0, 1]** by dividing by 255.
- **Split** the dataset into:
  - **80% Training** set
  - **20% Testing** set
- Used `ImageDataGenerator` from Keras with `validation_split` for train/test splitting.

### 3. Model Development
- Built and trained a CNN model using the Sequential API.
- Compiled with **Adam** optimizer, **Binary Crossentropy** loss, and **Accuracy** metric.
- Trained for **10 epochs**.

### 4. Model Evaluation
- Evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- Generated a **Confusion Matrix** heatmap.
- Plotted **Accuracy vs Epoch** and **Loss vs Epoch** training curves.

##  CNN Architecture

```
┌─────────────────────────────────────────────┐
│         Input: 128 × 128 × 3 (RGB)         │
├─────────────────────────────────────────────┤
│  Conv2D: 32 filters, 3×3 kernel, ReLU       │
│  MaxPooling2D: 2×2                           │
├─────────────────────────────────────────────┤
│  Conv2D: 64 filters, 3×3 kernel, ReLU       │
│  MaxPooling2D: 2×2                           │
├─────────────────────────────────────────────┤
│  Conv2D: 128 filters, 3×3 kernel, ReLU      │
│  MaxPooling2D: 2×2                           │
├─────────────────────────────────────────────┤
│  Flatten                                     │
├─────────────────────────────────────────────┤
│  Dense: 128 neurons, ReLU                    │
├─────────────────────────────────────────────┤
│  Dense: 1 neuron, Sigmoid (Output)           │
└─────────────────────────────────────────────┘
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D (32 filters) | (126, 126, 32) | 896 |
| MaxPooling2D | (63, 63, 32) | 0 |
| Conv2D (64 filters) | (61, 61, 64) | 18,496 |
| MaxPooling2D | (30, 30, 64) | 0 |
| Conv2D (128 filters) | (28, 28, 128) | 73,856 |
| MaxPooling2D | (14, 14, 128) | 0 |
| Flatten | (25,088) | 0 |
| Dense (128) | (128) | 3,211,392 |
| Dense (1) | (1) | 129 |

**Total Trainable Parameters:** ~3,304,769

## Results

### Evaluation Metrics

| Metric | Score |
|---|---|
| Test Accuracy | 91.48% |
| Test Loss | 0.2775 |
| Training Accuracy | 97.93% |
| Weighted Precision | 0.91 |
| Weighted Recall | 0.91 |
| Weighted F1-Score | 0.91 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cat | 0.95 | 0.96 | 0.95 | 2499 |
| Dog | 0.28 | 0.24 | 0.26 | 166 |
| **Weighted Avg** | **0.91** | **0.91** | **0.91** | **2665** |

### Visualizations

After running the script, the following plots are generated:

1. **`sample_images.png`** -- 5 sample images with class labels
2. **`confusion_matrix.png`** -- Heatmap of True vs Predicted labels
3. **`training_curves.png`** -- Accuracy and Loss curves over epochs

### Observations

1. The model achieved a training accuracy of 97.93% and a test accuracy of 91.48% after 10 epochs.
2. Training loss decreased from 0.2423 to 0.0617, showing effective feature learning.
3. The train-test accuracy gap is small (6.4%), indicating good generalization without significant overfitting.
4. The dataset is imbalanced (2499 Cat vs 166 Dog in the test set), which affects per-class Dog metrics. Data augmentation or class balancing could improve Dog-class recall.

## Conclusion

A Convolutional Neural Network was successfully developed to classify cat and dog images, achieving 91.48% test accuracy. **Convolution layers** automatically extract edges, textures, and patterns, while **pooling layers** reduce spatial dimensions and introduce translational invariance. A key **advantage of CNN over ANN** for image classification is parameter sharing through convolutional filters, which preserves spatial relationships and drastically reduces the number of trainable parameters. However, a **limitation of CNNs** is their dependence on large labeled datasets and significant computational resources (GPU) for training, which can be challenging in resource-constrained environments.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/shubham148-st/Image-Classification-using-CNN-MPOnline.git
   cd Image-Classification-using-CNN-MPOnline
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) and extract it into a `dataset/` folder.

3. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
   ```

4. Run the script:
   ```bash
   python Assignment-9.py
   ```
