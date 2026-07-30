
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# Configuration
# ============================================================================
# Update this path to point to the extracted dataset folder.
# Expected structure:
#   DATASET_DIR/
#       cats/   (or cat/)   — contains cat images
#       dogs/   (or dog/)   — contains dog images
#
# The Kaggle dataset "Dog and Cat Classification Dataset" typically
# has subfolders named "cats" and "dogs" (or similar).

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.20  # 80% train, 20% test


# Data Understanding 

print("=" * 70)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 70)

# 1.1 — Display folder structure
print("\n📁 Dataset Folder Structure:")
for root, dirs, files in os.walk(DATASET_DIR):
    level = root.replace(DATASET_DIR, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}📂 {os.path.basename(root)}/")
    # Only show up to 5 file names per folder to keep output clean
    sub_indent = "  " * (level + 1)
    shown = files[:5]
    for f in shown:
        print(f"{sub_indent}📄 {f}")
    if len(files) > 5:
        print(f"{sub_indent}... and {len(files) - 5} more files")

# 1.2 — Identify classes, image dimensions, total images
classes = sorted(
    [
        d
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
)
num_classes = len(classes)

total_images = 0
class_counts = {}
sample_dims = None

for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    imgs = [
        f
        for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    class_counts[cls] = len(imgs)
    total_images += len(imgs)

    # Read one sample to get original dimensions
    if sample_dims is None and len(imgs) > 0:
        sample_img = tf.keras.utils.load_img(os.path.join(cls_path, imgs[0]))
        sample_dims = np.array(sample_img).shape

print(f"\n📊 Number of classes : {num_classes}")
print(f"   Classes           : {classes}")
print(f"   Total images      : {total_images}")
for cls, count in class_counts.items():
    print(f"   Images in '{cls}'  : {count}")
if sample_dims is not None:
    print(f"   Sample image dims : {sample_dims}  (H × W × C)")

# 1.3 — Display 5 sample images with labels
print("\n🖼  Displaying 5 sample images with class labels...")
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")

sample_paths = []
for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    imgs = [
        f
        for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    chosen = random.sample(imgs, min(3, len(imgs)))
    for img_name in chosen:
        sample_paths.append((os.path.join(cls_path, img_name), cls))

random.shuffle(sample_paths)
sample_paths = sample_paths[:5]

for ax, (img_path, label) in zip(axes, sample_paths):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    ax.imshow(img)
    ax.set_title(label.capitalize(), fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved → sample_images.png")

# ============================================================================
# Task 2: Data Preprocessing (2 Marks)
# ============================================================================
print("\n" + "=" * 70)
print("TASK 2: DATA PREPROCESSING")
print("=" * 70)

# Resize to 128×128 and normalize pixel values to [0, 1].
# Split 80% training / 20% validation (testing) using validation_split.

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values to 0–1
    validation_split=VALIDATION_SPLIT,
)

print(f"\n🔧 Image size     : {IMG_SIZE}")
print(f"   Normalization  : pixel / 255  →  [0, 1]")
print(f"   Train split    : {100 * (1 - VALIDATION_SPLIT):.0f}%")
print(f"   Test split     : {100 * VALIDATION_SPLIT:.0f}%")
print(f"   Batch size     : {BATCH_SIZE}")

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42,
    shuffle=True,
)

test_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42,
    shuffle=False,
)

print(f"\n   Training samples   : {train_generator.samples}")
print(f"   Testing samples    : {test_generator.samples}")
print(f"   Class indices      : {train_generator.class_indices}")

# ============================================================================
# Task 3: Model Development (3 Marks)
# ============================================================================
print("\n" + "=" * 70)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 70)

# Build CNN architecture as specified
model = Sequential(
    [
        # Block 1
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        # Block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Block 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Classifier head
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),  # Binary output
    ]
)

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\n📐 Model Architecture:")
model.summary()

# Train for 10 epochs
print(f"\n🚀 Training for {EPOCHS} epochs ...\n")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1,
)

# ============================================================================
# Task 4: Model Evaluation (2 Marks)
# ============================================================================
print("\n" + "=" * 70)
print("TASK 4: MODEL EVALUATION")
print("=" * 70)

# --- 4.1  Evaluate on test set ---
test_generator.reset()
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\n📈 Test Loss     : {test_loss:.4f}")
print(f"   Test Accuracy : {test_accuracy:.4f}  ({test_accuracy * 100:.2f}%)")

# --- 4.2  Predictions for detailed metrics ---
test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\n   Precision : {precision:.4f}")
print(f"   Recall    : {recall:.4f}")
print(f"   F1-Score  : {f1:.4f}")

class_labels = list(train_generator.class_indices.keys())
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# --- 4.3  Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved → confusion_matrix.png")

# --- 4.4  Accuracy vs Epoch ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Test Accuracy", marker="s")
plt.title("Accuracy vs Epoch", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# --- 4.5  Loss vs Epoch ---
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", marker="o", color="red")
plt.plot(
    history.history["val_loss"], label="Test Loss", marker="s", color="orange"
)
plt.title("Loss vs Epoch", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("   ✅ Saved → training_curves.png")

# --- 4.6  Observations ---
print("\n📝 Observations:")
print("-" * 50)

final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]

print(
    f"1. The model achieved a training accuracy of {final_train_acc*100:.2f}% "
    f"and a test accuracy of {final_val_acc*100:.2f}% after {EPOCHS} epochs."
)
print(
    f"2. The training loss decreased from "
    f"{history.history['loss'][0]:.4f} to {final_train_loss:.4f}, "
    f"indicating that the model successfully learned discriminative "
    f"features from the cat and dog images."
)
gap = abs(final_train_acc - final_val_acc)
if gap > 0.10:
    print(
        f"3. There is a noticeable gap ({gap*100:.1f}%) between training and "
        f"test accuracy, suggesting some degree of overfitting. Data "
        f"augmentation or dropout regularization could help reduce this gap."
    )
else:
    print(
        f"3. The gap between training and test accuracy is small ({gap*100:.1f}%), "
        f"indicating that the model generalizes well and is not significantly "
        f"overfitting on the training data."
    )
print(
    f"4. The confusion matrix shows the distribution of correct and "
    f"incorrect predictions across both classes. A balanced performance "
    f"on cats and dogs indicates that the model does not exhibit strong "
    f"class bias."
)

# ============================================================================
# Task 5: Conclusion (1 Mark)
# ============================================================================
print("\n" + "=" * 70)
print("TASK 5: CONCLUSION")
print("=" * 70)

conclusion = f"""
In this assignment, a Convolutional Neural Network (CNN) was developed to
classify images of cats and dogs. The model achieved a test accuracy of
approximately {final_val_acc*100:.1f}%, demonstrating that CNNs can effectively
learn spatial hierarchies and discriminative features from raw pixel data.
Convolution layers act as automatic feature extractors — detecting edges,
textures, and shapes — while pooling layers reduce spatial dimensions and
provide translational invariance, making the model robust to small shifts in
the input. A key advantage of CNNs over traditional Artificial Neural Networks
(ANNs) is parameter sharing through convolutional filters, which dramatically
reduces the number of trainable parameters and preserves spatial relationships
in images. However, one limitation of CNNs is that they require large amounts
of labeled training data and significant computational resources (GPU) for
training, which can be a constraint in resource-limited environments.
"""

print(conclusion)

# Save the model
model.save("cats_vs_dogs_cnn_model.h5")
print("💾 Model saved → cats_vs_dogs_cnn_model.h5")
print("\n✅ Assignment 9 Complete!")
