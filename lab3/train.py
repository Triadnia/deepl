import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

import config
import plots
from data import build_split, check_split, eval_transform, make_train_loader, read_split, train_transform, write_split
from model import (build_model, count_params, criterion, frozen_params_in_optimizer, get_device,
                   optimizer_groups, set_train_mode, train_run, train_step)

BN_STATS = ("running_mean", "running_var")
BN_PROBE = "layer1.0.bn1"
LOG_NAME = "run.json"
LOGITS_NAME = "val_logits.npy"
WEIGHTS_NAME = "weights.pt"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=config.LAB_DIR, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".") if commit else None
    return {"commit": commit, "dirty": bool(status) if commit else None}


def file_sha256(path: Path = config.SPLIT_CSV) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()}, {platform.platform()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def environment(device: torch.device) -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "device_name": device_name(device),
    }


def yes_no(flag: bool) -> str:
    return "так" if flag else "ні"


def check_batch_figure(data_root: Path) -> None:
    torch.manual_seed(config.CHECK_SEED)
    images, labels = next(iter(make_train_loader(data_root)))
    plots.plot_batch(images, labels, config.FIGURES_DIR / "batch.png")


def check_initial_loss(data_root: Path, device: torch.device) -> dict:
    results = {"ln6": math.log(config.NUM_CLASSES)}
    for regime in ("scratch", "frozen"):
        model, _ = build_model(regime, config.CHECK_SEED, device)
        images, labels = next(iter(make_train_loader(data_root)))
        set_train_mode(model, regime)
        with torch.no_grad():
            loss = criterion(model(images.to(device)), labels.to(device)).item()
        results[regime] = loss
        print(f"L0 {regime:<8} = {loss:.4f}   (ln 6 = {results['ln6']:.4f}, "
              f"відхилення {loss - results['ln6']:+.4f})")
    return results


def check_params(device: torch.device) -> tuple[list[dict], bool]:
    rows, all_ok = [], True
    for regime, cfg in config.REGIMES.items():
        model, optimizer = build_model(regime, config.CHECK_SEED, device)
        total, trainable = count_params(model)
        groups = optimizer_groups(optimizer)
        leaked = frozen_params_in_optimizer(optimizer)
        ok = (total == config.TOTAL_PARAMS and trainable == cfg["trainable_params"]
              and sum(n for n, _ in groups) == trainable and leaked == 0)
        all_ok &= ok
        rows.append({"regime": regime, "total": total, "trainable": trainable,
                     "groups": [{"params": n, "lr": lr} for n, lr in groups],
                     "frozen_in_optimizer": leaked, "matches": ok})
        groups_text = "; ".join(f"{n:,} @ {lr:g}".replace(",", " ") for n, lr in groups)
        print(f"{regime:<9} усього {total:,}  навчається {trainable:,}  групи: {groups_text}  "
              f"заморожених в оптимізаторі: {leaked}  -> {yes_no(ok)}".replace(",", " "))
    return rows, all_ok


def run_steps_and_compare(regime: str, data_root: Path, device: torch.device,
                          use_set_train_mode: bool = True):
    model, optimizer = build_model(regime, config.CHECK_SEED, device)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    param_names = {name for name, _ in model.named_parameters()}

    if use_set_train_mode:
        set_train_mode(model, regime)
    else:
        model.train()
    loader_iter = iter(make_train_loader(data_root))
    for _ in range(config.CHECK_STEPS):
        images, labels = next(loader_iter)
        train_step(model, optimizer, images, labels, device)

    after = model.state_dict()
    eval_modules = config.REGIMES[regime]["eval_modules"]
    rows, all_ok = [], True
    for module in config.TOP_MODULES:
        keys = [k for k in before if k.startswith(module + ".")]
        p_keys = [k for k in keys if k in param_names]
        bn_keys = [k for k in keys if k.endswith(BN_STATS)]
        params_changed = any(not torch.equal(before[k], after[k]) for k in p_keys)
        bn_changed = any(not torch.equal(before[k], after[k]) for k in bn_keys) if bn_keys else None

        frozen = module in eval_modules
        expected_bn = None if not bn_keys else not frozen
        ok = params_changed == (not frozen) and bn_changed == expected_bn
        all_ok &= ok
        rows.append({"module": module, "params_changed": params_changed,
                     "bn_changed": bn_changed, "matches": ok})

    probe = {stat: (after[f"{BN_PROBE}.{stat}"] - before[f"{BN_PROBE}.{stat}"]).abs().max().item()
             for stat in BN_STATS}
    return rows, all_ok, probe


def print_change_table(regime: str, rows: list[dict]) -> None:
    print(f"{'Режим':<9} {'Модуль':<8} {'Параметри':>10} {'BN-статистики':>14} {'Очікувано':>10}")
    for r in rows:
        bn = "немає BN" if r["bn_changed"] is None else yes_no(r["bn_changed"])
        print(f"{regime:<9} {r['module']:<8} {yes_no(r['params_changed']):>10} {bn:>14} "
              f"{yes_no(r['matches']):>10}")


def check_changes(data_root: Path, device: torch.device) -> tuple[dict, bool]:
    results, all_ok = {}, True
    for regime in ("frozen", "unfreeze"):
        rows, ok, _ = run_steps_and_compare(regime, data_root, device)
        print_change_table(regime, rows)
        results[regime] = rows
        all_ok &= ok
    return results, all_ok


def bn_experiment(data_root: Path, device: torch.device) -> dict:
    print("Прогноз: параметри замороженого кістяка не зміняться (requires_grad=False), але running_mean/"
          "running_var усіх BN кістяка зміняться, бо в режимі train() BN оновлює їх під час прямого "
          "проходу. На кривій втрати train цього майже не видно, бо втрата рахується в train-режимі "
          "за статистиками пакета; розбіжність проявиться в eval() на validation.")
    rows, _, probe = run_steps_and_compare("frozen", data_root, device, use_set_train_mode=False)
    print_change_table("frozen*", rows)
    print(f"{BN_PROBE}: max|Δ running_mean| = {probe['running_mean']:.6f}, "
          f"max|Δ running_var| = {probe['running_var']:.6f}")
    return {"rows": rows, "layer1.0.bn1": probe}


def run_all_checks(data_root: Path, device: torch.device) -> bool:
    print("\n=== 1. Поділ ===")
    split_ok = check_split(read_split())

    print("\n=== 1. Пакет після перетворень ===")
    check_batch_figure(data_root)

    print("\n=== 1. Початкова втрата ===")
    initial_loss = check_initial_loss(data_root, device)

    print("\n=== 2. Параметри й групи оптимізатора ===")
    params, params_ok = check_params(device)

    print("\n=== 2. Що змінюється за 5 кроків ===")
    changes, changes_ok = check_changes(data_root, device)

    print("\n=== 2. Дослід із навмисною помилкою (frozen + model.train()) ===")
    experiment = bn_experiment(data_root, device)

    all_ok = split_ok and params_ok and changes_ok
    results = {"device": str(device), "split_ok": split_ok, "initial_loss": initial_loss,
               "params": params, "changes": changes, "bn_experiment": experiment,
               "all_ok": all_ok}
    with open(config.LAB_DIR / "checks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nУсі перевірки збіглися з очікуваним: {yes_no(all_ok)}")
    return all_ok


def run_dir(regime: str, seed: int) -> Path:
    return config.RUNS_DIR / f"{regime}_seed{seed}"


def run_settings(regime: str, result: dict) -> dict:
    model, optimizer = result["model"], result["optimizer"]
    train_loader, val_loader = result["train_loader"], result["val_loader"]
    cfg = config.REGIMES[regime]
    groups = [{"modules": names, "numel": sum(p.numel() for p in g["params"]),
               **{k: v for k, v in g.items() if k != "params"}}
              for (names, _), g in zip(cfg["groups"], optimizer.param_groups)]
    return {
        "transforms": {
            "train": repr(train_transform),
            "validation_test": repr(eval_transform),
        },
        "model": {
            "architecture": "torchvision.models.resnet18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1" if cfg["pretrained"] else None,
            "fc": repr(model.fc),
            "trainable": cfg["trainable"],
            "eval_modules": cfg["eval_modules"],
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batchnorm": {"momentum": model.bn1.momentum, "eps": model.bn1.eps},
            "dtype": str(next(model.parameters()).dtype),
        },
        "optimizer": {"class": type(optimizer).__name__, "groups": groups},
        "loss": repr(criterion) + ", reduction=" + criterion.reduction,
        "loop": {
            "epochs": config.EPOCHS,
            "train_loader": {"batch_size": train_loader.batch_size,
                             "shuffle": isinstance(train_loader.sampler,
                                                   torch.utils.data.RandomSampler),
                             "drop_last": train_loader.drop_last,
                             "num_workers": train_loader.num_workers,
                             "generator": None},
            "eval_loader": {"batch_size": val_loader.batch_size, "shuffle": False,
                            "num_workers": val_loader.num_workers},
            "steps": config.EPOCHS * len(train_loader),
            "lr_schedule": None, "grad_clipping": None, "mixed_precision": False,
            "label_smoothing": criterion.label_smoothing, "class_weights": None,
            "early_stopping": None, "checkpoint_selection": "останній (після 8-ї епохи)",
        },
    }


def save_run(regime: str, seed: int, result: dict, device: torch.device) -> dict:
    out = run_dir(regime, seed)
    out.mkdir(parents=True, exist_ok=True)
    log = {
        "regime": regime,
        "seed": seed,
        "settings": run_settings(regime, result),
        "code": git_info(),
        "split_sha256": file_sha256(config.SPLIT_CSV),
        "environment": environment(device),
        "history": result["history"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "final_val_f1": result["history"]["val_f1"][-1],
        "train_time_sec": result["train_time_sec"],
    }
    with open(out / LOG_NAME, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    np.save(out / LOGITS_NAME, result["val_logits"].astype(np.float32))
    state_dict = {k: v.detach().cpu() for k, v in result["model"].state_dict().items()}
    torch.save(state_dict, out / WEIGHTS_NAME)
    return log


def run_experiment(data_root: Path, device: torch.device) -> list[dict]:
    logs = []
    for i, (regime, seed) in enumerate(config.RUNS, start=1):
        print(f"\n--- Запуск {i}/{len(config.RUNS)}: {regime}, зерно {seed} ---")
        result = train_run(regime, seed, data_root, device)
        logs.append(save_run(regime, seed, result, device))
        print(f"  час навчання {result['train_time_sec']:.1f} с")
    return logs


def load_logs() -> list[dict]:
    logs = []
    for regime, seed in config.RUNS:
        path = run_dir(regime, seed) / LOG_NAME
        if not path.exists():
            raise SystemExit(f"Немає журналу {path.relative_to(config.LAB_DIR)}: спершу запустіть train.py")
        with open(path, encoding="utf-8") as f:
            logs.append(json.load(f))
    return logs


def load_val_logits(regime: str, seed: int) -> np.ndarray:
    return np.load(run_dir(regime, seed) / LOGITS_NAME)


def format_runs_table(logs: list[dict]) -> str:
    lines = ["| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |",
             "|---|---|---|---|---|---|"]
    for i, log in enumerate(logs, start=1):
        lines.append(f"| {i} | {log['regime']} | {log['seed']} | {log['final_train_loss']:.4f} | "
                     f"{log['final_val_f1']:.4f} | {log['train_time_sec']:.1f} |")
    return "\n".join(lines)


def regime_scores(logs: list[dict]) -> dict[str, list[float]]:
    scores = {}
    for log in logs:
        scores.setdefault(log["regime"], []).append(log["final_val_f1"])
    return scores


def select(logs: list[dict]) -> tuple[str, int]:
    scores = regime_scores(logs)
    regime = max(scores, key=lambda r: np.mean(scores[r]))
    best = max((log for log in logs if log["regime"] == regime), key=lambda log: log["final_val_f1"])
    return regime, best["seed"]


def save_selected_model(regime: str, seed: int) -> None:
    state_dict = torch.load(run_dir(regime, seed) / WEIGHTS_NAME, map_location="cpu")
    torch.save({
        "state_dict": state_dict,
        "regime": regime,
        "seed": seed,
        "classes": config.CLASSES,
        "img_size": config.IMG_SIZE,
    }, config.MODEL_PATH)
    print(f"Збережено {config.MODEL_PATH.name}: {regime}, зерно {seed}")


def load_selected_model() -> dict:
    return torch.load(config.MODEL_PATH, map_location="cpu")


def verify_selection(logs: list[dict], checkpoint: dict) -> tuple[str, int]:
    regime, seed = select(logs)
    if (checkpoint["regime"], checkpoint["seed"]) != (regime, seed):
        raise SystemExit(f"{config.MODEL_PATH.name} містить {checkpoint['regime']}/{checkpoint['seed']}, "
                         f"а правило вибору за журналами дає {regime}/{seed}")
    if checkpoint["classes"] != config.CLASSES or checkpoint["img_size"] != config.IMG_SIZE:
        raise SystemExit(f"{config.MODEL_PATH.name}: classes або img_size не збігаються з config")
    return regime, seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Команда навчання: поділ, перевірки пунктів 1-2, сім запусків, вибір, lab3_model.pt")
    parser.add_argument("--data", type=Path, required=True,
                        help="тека з розпакованим набором (містить seg_train/ і seg_test/)")
    args = parser.parse_args()
    device = get_device()
    print(f"Пристрій: {device}")

    write_split(build_split(args.data))
    print(f"Записано {config.SPLIT_CSV.name}")

    if not run_all_checks(args.data, device):
        raise SystemExit("Перевірки пунктів 1-2 не збіглися з очікуваним: основний експеримент не запускається")

    print("\n=== 3. Сім запусків ===")
    logs = run_experiment(args.data, device)
    print("\n" + format_runs_table(logs))

    print("\n=== 5. Вибір конфігурації за validation ===")
    for regime, scores in regime_scores(logs).items():
        print(f"{regime:<9} F̄ = {sum(scores) / len(scores):.4f}  ({', '.join(f'{s:.4f}' for s in scores)})")
    regime, seed = select(logs)
    save_selected_model(regime, seed)


if __name__ == "__main__":
    main()
