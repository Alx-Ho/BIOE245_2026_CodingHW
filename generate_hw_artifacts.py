import json
import os

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import seaborn as sns
from medmnist import INFO
from sklearn.metrics import auc, roc_curve
from tensorboard.backend.event_processing import event_accumulator

RUN_DIR = "/scratch/hw/pathmnist/260323_091229"
DATA_NPZ = "/data/pathmnist/pathmnist.npz"
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# Load dataset arrays for counts and image extraction
arr = np.load(DATA_NPZ)
train_images, train_labels = arr["train_images"], arr["train_labels"].reshape(-1)
val_images, val_labels = arr["val_images"], arr["val_labels"].reshape(-1)
test_images, test_labels = arr["test_images"], arr["test_labels"].reshape(-1)

# Load test predictions CSV (col0=sample index, col1..=class probs)
test_csv = os.path.join(RUN_DIR, "pathmnist_test_[AUC]0.985_[ACC]0.895@model1.csv")
df_test = pd.read_csv(test_csv, header=None)
y_true = test_labels.astype(int)
y_score = df_test.iloc[:, 1:].to_numpy(dtype=float)
y_pred = np.argmax(y_score, axis=1)

# Parse TensorBoard scalar curves
event_path = os.path.join(RUN_DIR, "Tensorboard_Results", "events.out.tfevents.1774282350.dillon")
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()
scalar_tags = ea.Tags().get("scalars", [])

# Build scalar table
tag_to_vals = {}
for tag in scalar_tags:
    scalars = ea.Scalars(tag)
    tag_to_vals[tag] = {
        "steps": [s.step for s in scalars],
        "values": [s.value for s in scalars],
    }

# Plot training/stat curves
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
axes = axes.ravel()
for i, tag in enumerate(sorted(scalar_tags)):
    ax = axes[i]
    steps = tag_to_vals[tag]["steps"]
    values = tag_to_vals[tag]["values"]
    ax.plot(steps, values, linewidth=1.6)
    ax.set_title(tag)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
training_plot = os.path.join(OUT_DIR, "training_curves.png")
plt.savefig(training_plot, dpi=180)
plt.close(fig)

# ROC/AUC figure for multi-class one-vs-rest
n_classes = y_score.shape[1]
y_bin = np.eye(n_classes)[y_true]

fpr = {}
tpr = {}
roc_auc = {}
for c in range(n_classes):
    fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_score[:, c])
    roc_auc[c] = auc(fpr[c], tpr[c])

fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for c in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
mean_tpr /= n_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

fig = plt.figure(figsize=(10, 8))
plt.plot(fpr_micro, tpr_micro, label=f"micro-average ROC (AUC = {roc_auc_micro:.3f})", linewidth=2.2)
plt.plot(all_fpr, mean_tpr, label=f"macro-average ROC (AUC = {roc_auc_macro:.3f})", linewidth=2.2)
for c in [0, 1, 2, 3, 4]:
    plt.plot(fpr[c], tpr[c], alpha=0.6, linewidth=1.4, label=f"class {c} ROC (AUC = {roc_auc[c]:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1.2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("PathMNIST Test ROC Curves (one-vs-rest)")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
roc_plot = os.path.join(OUT_DIR, "test_roc_curves.png")
plt.savefig(roc_plot, dpi=180)
plt.close(fig)

# Class names from MedMNIST info
labels_dict = INFO["pathmnist"]["label"]
class_names = {int(k): v for k, v in labels_dict.items()}

# Select one correct + one incorrect sample for five classes
selected = {}
candidate_classes = []
for c in range(n_classes):
    idx_true = np.where(y_true == c)[0]
    correct = idx_true[y_pred[idx_true] == c]
    wrong = idx_true[y_pred[idx_true] != c]
    if len(correct) > 0 and len(wrong) > 0:
        candidate_classes.append(c)
        selected[c] = {"correct_idx": int(correct[0]), "wrong_idx": int(wrong[0])}

selected_classes = candidate_classes[:5]
example_image_files = []

# Plot 10 images (5 classes x correct/incorrect)
fig, axes = plt.subplots(len(selected_classes), 2, figsize=(8, 3.6 * len(selected_classes)))
if len(selected_classes) == 1:
    axes = np.array([axes])
for row, c in enumerate(selected_classes):
    c_name = class_names.get(c, str(c))
    info = selected[c]

    ci = info["correct_idx"]
    wi = info["wrong_idx"]

    cimg = test_images[ci]
    wimg = test_images[wi]

    c_file = os.path.join(OUT_DIR, f"class_{c}_correct_idx_{ci}.png")
    w_file = os.path.join(OUT_DIR, f"class_{c}_wrong_idx_{wi}.png")
    plt.imsave(c_file, cimg)
    plt.imsave(w_file, wimg)
    example_image_files.extend([c_file, w_file])

    axes[row, 0].imshow(cimg)
    axes[row, 0].axis("off")
    axes[row, 0].set_title(
        f"Class {c}: {c_name} | correct\\ntrue={y_true[ci]}, pred={y_pred[ci]}",
        fontsize=10,
    )

    axes[row, 1].imshow(wimg)
    axes[row, 1].axis("off")
    axes[row, 1].set_title(
        f"Class {c}: {c_name} | incorrect\\ntrue={y_true[wi]}, pred={y_pred[wi]}",
        fontsize=10,
    )

plt.tight_layout()
examples_plot = os.path.join(OUT_DIR, "class_examples_correct_incorrect.png")
plt.savefig(examples_plot, dpi=180)
plt.close(fig)

# Save summary JSON for markdown authoring
summary = {
    "run_dir": RUN_DIR,
    "dataset_npz": DATA_NPZ,
    "split_counts": {
        "train": int(train_images.shape[0]),
        "val": int(val_images.shape[0]),
        "test": int(test_images.shape[0]),
    },
    "image_shape": list(train_images.shape[1:]),
    "n_classes": int(n_classes),
    "scalar_tags": sorted(scalar_tags),
    "num_scalar_curves": len(scalar_tags),
    "test_accuracy_from_csv": float((y_pred == y_true).mean()),
    "roc_auc_micro": float(roc_auc_micro),
    "roc_auc_macro": float(roc_auc_macro),
    "selected_examples": selected,
    "selected_classes": selected_classes,
    "class_names": {str(k): v for k, v in class_names.items()},
    "generated_figures": [
        training_plot,
        roc_plot,
        examples_plot,
    ],
    "example_image_files": example_image_files,
}

summary_path = os.path.join(OUT_DIR, "analysis_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
