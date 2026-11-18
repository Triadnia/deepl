import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

import config
from data import make_train_loader, make_val_loader

criterion = nn.CrossEntropyLoss()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _modules(model: nn.Module, names: list[str]) -> list[nn.Module]:
    if names == ["*"]:
        return [model]
    return [getattr(model, name) for name in names]


def build_model(regime: str, seed: int, device: torch.device):
    cfg = config.REGIMES[regime]

    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    model = resnet18(weights=weights)

    model.fc = nn.Linear(512, config.NUM_CLASSES)

    for p in model.parameters():
        p.requires_grad = False
    for module in _modules(model, cfg["trainable"]):
        for p in module.parameters():
            p.requires_grad = True

    model.to(device)
    return model, make_optimizer(model, regime)


def set_train_mode(model: nn.Module, regime: str) -> None:
    model.train()
    for module in _modules(model, config.REGIMES[regime]["eval_modules"]):
        module.eval()


def make_optimizer(model: nn.Module, regime: str) -> torch.optim.AdamW:
    groups = []
    for names, lr in config.REGIMES[regime]["groups"]:
        params = [p for module in _modules(model, names)
                  for p in module.parameters() if p.requires_grad]
        groups.append({"params": params, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def optimizer_groups(optimizer: torch.optim.Optimizer) -> list[tuple[int, float]]:
    return [(sum(p.numel() for p in g["params"]), g["lr"]) for g in optimizer.param_groups]


def frozen_params_in_optimizer(optimizer: torch.optim.Optimizer) -> int:
    return sum(1 for g in optimizer.param_groups for p in g["params"] if not p.requires_grad)


def train_step(model, optimizer, images, labels, device) -> float:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(model, regime, loader, optimizer, device) -> float:
    set_train_mode(model, regime)
    total, count = 0.0, 0
    for images, labels in loader:
        total += train_step(model, optimizer, images, labels, device) * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for images, y in loader:
        logits.append(model(images.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, logits.argmax(axis=1), average="macro",
                          labels=list(range(config.NUM_CLASSES))))


def train_run(regime: str, seed: int, data_root: Path, device: torch.device) -> dict:
    model, optimizer = build_model(regime, seed, device)
    train_loader = make_train_loader(data_root)
    val_loader = make_val_loader(data_root)

    history = {"train_loss": [], "val_f1": []}
    start = time.perf_counter()
    for epoch in range(1, config.EPOCHS + 1):
        loss = train_one_epoch(model, regime, train_loader, optimizer, device)
        val_logits, val_labels = predict(model, val_loader, device)
        f1 = macro_f1(val_logits, val_labels)
        history["train_loss"].append(loss)
        history["val_f1"].append(f1)
        print(f"  {regime} seed {seed} | епоха {epoch}: втрата {loss:.4f}, macro-F1 {f1:.4f}")
    train_time = time.perf_counter() - start

    return {
        "model": model,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "history": history,
        "val_logits": val_logits,
        "train_time_sec": train_time,
    }
