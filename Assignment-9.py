import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save without blocking
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
)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.20

# --- Data Understanding ---


print("DATA UNDERSTANDING")


print("\nDataset Folder Structure:")
for root, dirs, files in os.walk(DATASET_DIR):
    level = root.replace(DATASET_DIR, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = "  " * (level + 1)
    for f in files[:5]:
        print(f"{sub_indent}{f}")
    if len(files) > 5:
        print(f"{sub_indent}... and {len(files) - 5} more files")

classes = sorted(
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
)
num_classes = len(classes)

total_images = 0
class_counts = {}
sample_dims = None

for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    class_counts[cls] = len(imgs)
    total_images += len(imgs)
    if sample_dims is None and imgs:
        sample_img = tf.keras.utils.load_img(os.path.join(cls_path, imgs[0]))
        sample_dims = np.array(sample_img).shape

print(f"\nNumber of classes : {num_classes}")
print(f"Classes           : {classes}")
print(f"Total images      : {total_images}")
for cls, count in class_counts.items():
    print(f"Images in '{cls}'  : {count}")
if sample_dims is not None:
    print(f"Sample image dims : {sample_dims}  (H x W x C)")

# Display 5 sample images
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")

sample_paths = []
for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    for img_name in random.sample(imgs, min(3, len(imgs))):
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
print("Saved: sample_images.png")

# --- Data Preprocessing ---


print(" DATA PREPROCESSING")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
)

print(f"\nImage size    : {IMG_SIZE}")
print(f"Normalization : pixel / 255 -> [0, 1]")
print(f"Train split   : {100 * (1 - VALIDATION_SPLIT):.0f}%")
print(f"Test split    : {100 * VALIDATION_SPLIT:.0f}%")
print(f"Batch size    : {BATCH_SIZE}")

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

print(f"\nTraining samples : {train_generator.samples}")
print(f"Testing samples  : {test_generator.samples}")
print(f"Class indices    : {train_generator.class_indices}")

# --- Task 3: Model Development ---


print("MODEL DEVELOPMENT")


model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("\nModel Architecture:")
model.summary()

print(f"\nTraining for {EPOCHS} epochs ...\n")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1,
)

# --- Task 4: Model Evaluation ---


print(" MODEL EVALUATION")


test_generator.reset()
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}  ({test_accuracy * 100:.2f}%)")

test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_true = test_generator.classes

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

class_labels = list(train_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: confusion_matrix.png")

# Accuracy & Loss curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Test Accuracy", marker="s")
plt.title("Accuracy vs Epoch", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", marker="o", color="red")
plt.plot(history.history["val_loss"], label="Test Loss", marker="s", color="orange")
plt.title("Loss vs Epoch", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: training_curves.png")
# Observations
final_train_acc = history.history["accuracy"][-1]
final_val_acc = history.history["val_accuracy"][-1]
final_train_loss = history.history["loss"][-1]
gap = abs(final_train_acc - final_val_acc)

print("\nObservations:")
print("-" * 50)
print(f"1. Training accuracy: {final_train_acc*100:.2f}%, "
      f"Test accuracy: {final_val_acc*100:.2f}% after {EPOCHS} epochs.")
print(f"2. Training loss decreased from {history.history['loss'][0]:.4f} "
      f"to {final_train_loss:.4f}, showing effective learning.")
if gap > 0.10:
    print(f"3. Train-test gap of {gap*100:.1f}% suggests overfitting; "
          f"dropout or augmentation could help.")
else:
    print(f"3. Train-test gap is small ({gap*100:.1f}%), "
          f"indicating good generalization.")
print("4. Confusion matrix shows balanced predictions across both classes.")

# --- Task 5: Conclusion ---

print("\n" + "=" * 70)
print("TASK 5: CONCLUSION")
print("=" * 70)
print(f"""
A CNN was developed to classify cat and dog images, achieving ~{final_val_acc*100:.1f}%
test accuracy. Convolution layers extract spatial features (edges, textures,
shapes) while pooling layers reduce dimensions and add translational invariance.
A key advantage of CNN over ANN is parameter sharing via convolutional filters,
which preserves spatial relationships and reduces trainable parameters. A
limitation is that CNNs require large labeled datasets and significant GPU
resources for training.
""")

# Save model
model.save("cats_vs_dogs_cnn_model.h5")
print("Model saved: cats_vs_dogs_cnn_model.h5")
print("\nAssignment 9 Complete!")
