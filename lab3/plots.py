from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from data import denormalize


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Рисунок збережено: {path.relative_to(config.LAB_DIR)}")


def plot_batch(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    images = denormalize(images[:16].cpu())
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.6))
    for ax, image, label in zip(axes.flat, images, labels[:16]):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(config.CLASSES[int(label)], fontsize=10)
        ax.axis("off")
    fig.suptitle("Навчальний пакет після перетворень")
    _save(fig, path)


REGIME_COLORS = {"scratch": "tab:red", "frozen": "tab:blue", "unfreeze": "tab:green"}
SEED_STYLES = {1: "-", 2: "--", 3: ":"}


def plot_curves(logs: list[dict], path: Path) -> None:
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, config.EPOCHS + 1)
    for log in logs:
        style = {"color": REGIME_COLORS[log["regime"]], "linestyle": SEED_STYLES[log["seed"]],
                 "marker": "o", "markersize": 3, "label": f"{log['regime']}, зерно {log['seed']}"}
        ax_loss.plot(epochs, log["history"]["train_loss"], **style)
        ax_f1.plot(epochs, log["history"]["val_f1"], **style)
    ax_loss.set(title="Втрата на train", xlabel="Епоха", ylabel="CrossEntropy")
    ax_f1.set(title="macro-F1 на validation", xlabel="Епоха", ylabel="macro-F1")
    for ax in (ax_loss, ax_f1):
        ax.grid(alpha=0.3)
        ax.set_xticks(list(epochs))
    ax_f1.legend(fontsize=8, loc="center right")
    _save(fig, path)


def plot_confusion(matrix, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES), config.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(config.NUM_CLASSES), config.CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас", title=title)
    threshold = matrix.max() / 2
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_errors(errors: list[dict], path: Path) -> None:
    from PIL import Image

    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.8))
    for ax, e in zip(axes, errors):
        ax.imshow(Image.open(e["path"]).convert("RGB"))
        ax.set_title(f"{e['file']}\n{e['true']} → {e['pred']}\np = {e['prob']:.3f}", fontsize=9)
        ax.axis("off")
    _save(fig, path)
