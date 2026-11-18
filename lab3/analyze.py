import argparse
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision.models import resnet18

import config
import plots
from data import make_test_loader, make_val_loader
from model import macro_f1, predict
from train import (file_sha256, format_runs_table, load_logs, load_selected_model, load_val_logits,
                   regime_scores, verify_selection)

CPU = torch.device("cpu")
NUM_ERRORS = 5


def verdict(delta: float, noise: float) -> str:
    if abs(delta) <= noise:
        return "|Δ| ≤ R: різниця не відрізняється від розкиду, результату немає"
    return "|Δ| > R: ефект вартий перевірки, але не доведений"


def compute_effects(logs: list[dict]) -> dict:
    scores = regime_scores(logs)
    mean = {r: float(np.mean(s)) for r, s in scores.items()}
    spread = {r: float(np.max(s) - np.min(s)) for r, s in scores.items() if len(s) > 1}
    noise = max(spread["frozen"], spread["unfreeze"])
    f_scratch = scores["scratch"][0]
    d_pre = mean["frozen"] - f_scratch
    d_unf = mean["unfreeze"] - mean["frozen"]
    return {
        "F_scratch_1": f_scratch,
        "mean_frozen": mean["frozen"], "R_frozen": spread["frozen"],
        "mean_unfreeze": mean["unfreeze"], "R_unfreeze": spread["unfreeze"],
        "R": noise,
        "delta_pre": d_pre, "verdict_pre": verdict(d_pre, noise),
        "delta_unf": d_unf, "verdict_unf": verdict(d_unf, noise),
        "delta_total": d_pre + d_unf,
    }


def format_effects(e: dict) -> str:
    rows = [
        ("F_scratch,1", e["F_scratch_1"]),
        ("F̄_frozen", e["mean_frozen"]), ("R_frozen", e["R_frozen"]),
        ("F̄_unfreeze", e["mean_unfreeze"]), ("R_unfreeze", e["R_unfreeze"]),
        ("R = max(R_frozen, R_unfreeze)", e["R"]),
        ("Δpre = F̄_frozen − F_scratch,1", e["delta_pre"]),
        ("Δunf = F̄_unfreeze − F̄_frozen", e["delta_unf"]),
        ("Δpre + Δunf = F̄_unfreeze − F_scratch,1", e["delta_total"]),
    ]
    lines = ["| Величина | Значення |", "|---|---|"]
    lines += [f"| {name} | {value:.4f} |" for name, value in rows]
    lines += ["", f"Δpre: {e['verdict_pre']}", f"Δunf: {e['verdict_unf']}"]
    return "\n".join(lines)


def load_model_cpu(checkpoint: dict) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, config.NUM_CLASSES)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(CPU).eval()


def check_against_validation(model: nn.Module, data_root: Path, logged_logits: np.ndarray,
                             logged_f1: float) -> dict:
    logits, labels = predict(model, make_val_loader(data_root), CPU)
    agreement = float((logits.argmax(1) == logged_logits.argmax(1)).mean())
    f1 = macro_f1(logits, labels)
    print(f"Validation на CPU: macro-F1 {f1:.4f} (у журналі {logged_f1:.4f}), "
          f"збіг передбачень із val_logits.npy {agreement:.2%}")
    if agreement < 0.99:
        raise SystemExit("Ваги lab3_model.pt не відповідають збереженим логітам вибраного запуску")
    return {"val_f1_cpu": f1, "val_prediction_agreement": agreement}


def top_errors(matrix: np.ndarray, files: list[str], paths: list[Path], labels: np.ndarray,
               preds: np.ndarray, probs: np.ndarray) -> list[dict]:
    pairs = [(t, p) for t in range(config.NUM_CLASSES) for p in range(config.NUM_CLASSES)
             if t != p and matrix[t, p] > 0]
    pairs.sort(key=lambda tp: (-matrix[tp], tp[0], tp[1]))
    by_name = sorted(range(len(files)), key=lambda i: files[i])
    files_by_pair = {(t, p): [i for i in by_name if labels[i] == t and preds[i] == p]
                     for t, p in pairs}

    chosen, level = [], 0
    while len(chosen) < NUM_ERRORS:
        added = False
        for pair in pairs:
            if len(chosen) == NUM_ERRORS:
                break
            if len(files_by_pair[pair]) > level:
                chosen.append((pair, files_by_pair[pair][level]))
                added = True
        if not added:
            break
        level += 1

    return [{"file": files[i], "path": paths[i], "true": config.CLASSES[t],
             "pred": config.CLASSES[p], "prob": float(probs[i, p]), "count": int(matrix[t, p])}
            for (t, p), i in chosen]


def evaluate_test(model: nn.Module, data_root: Path) -> dict:
    loader = make_test_loader(data_root)
    logits, labels = predict(model, loader, CPU)
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    matrix = confusion_matrix(labels, preds, labels=list(range(config.NUM_CLASSES)))
    accuracy = float((preds == labels).mean())
    n = len(labels)
    dataset = loader.dataset
    return {
        "test_f1": macro_f1(logits, labels),
        "accuracy": accuracy,
        "n": n,
        "se_accuracy": math.sqrt(accuracy * (1 - accuracy) / n),
        "confusion": matrix,
        "errors": top_errors(matrix, dataset.files, dataset.paths, labels, preds, probs),
    }


def format_confusion(matrix: np.ndarray) -> str:
    lines = ["| справжній \\ передбачений | " + " | ".join(config.CLASSES) + " |",
             "|---" * (config.NUM_CLASSES + 1) + "|"]
    for i, name in enumerate(config.CLASSES):
        lines.append(f"| {name} | " + " | ".join(str(int(v)) for v in matrix[i]) + " |")
    return "\n".join(lines)


def format_errors(errors: list[dict]) -> str:
    lines = ["| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | "
             "Кількість таких помилок у test |", "|---|---|---|---|---|"]
    for e in errors:
        lines.append(f"| {e['file']} | {e['true']} | {e['pred']} | {e['prob']:.3f} | {e['count']} |")
    return "\n".join(lines)


def check_split_hash(logs: list[dict]) -> str:
    current = file_sha256(config.SPLIT_CSV)
    hashes = {log["split_sha256"] for log in logs}
    if hashes != {current}:
        raise SystemExit(f"SHA-256 split.csv у журналах ({', '.join(sorted(hashes))}) не збігається "
                         f"з поточним split.csv ({current}). Перенавчіть усі сім запусків.")
    return current


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Команда аналізу: таблиця, криві, ефекти, перевірка вибору, одноразова оцінка на test")
    parser.add_argument("--data", type=Path, required=True,
                        help="тека з розпакованим набором (містить seg_train/ і seg_test/)")
    args = parser.parse_args()

    report = []

    def out(text: str = "") -> None:
        print(text)
        report.append(text)

    logs = load_logs()
    split_hash = check_split_hash(logs)
    out(f"SHA-256 split.csv однаковий у всіх 7 журналах і збігається з поточним: `{split_hash}`")
    commits = {(log["code"]["commit"], log["code"]["dirty"]) for log in logs}
    out(f"Коміти коду в журналах: {sorted(commits, key=str)}")

    out("\n## 3. Сім запусків\n")
    out(format_runs_table(logs))
    plots.plot_curves(logs, config.FIGURES_DIR / "curves.png")

    out("\n## 4. Розклад ефектів\n")
    effects = compute_effects(logs)
    out(format_effects(effects))

    out("\n## 5. Вибір конфігурації і тестова оцінка\n")
    checkpoint = load_selected_model()
    regime, seed = verify_selection(logs, checkpoint)
    selected = next(log for log in logs if (log["regime"], log["seed"]) == (regime, seed))
    out(f"Вибрано: {regime}, зерно {seed} (F_r,s на validation = {selected['final_val_f1']:.4f}). "
        f"{config.MODEL_PATH.name} відповідає правилу вибору.")
    if abs(effects["delta_unf"]) <= effects["R"]:
        out("Увага: |Δunf| ≤ R, тож вибір між frozen і unfreeze визначив шум.")

    model = load_model_cpu(checkpoint)
    check_against_validation(model, args.data, load_val_logits(regime, seed), selected["final_val_f1"])

    test = evaluate_test(model, args.data)
    out(f"\nmacro-F1 на test: {test['test_f1']:.4f}")
    out(f"macro-F1 на validation вибраного запуску: {selected['final_val_f1']:.4f} "
        f"(різниця test − validation = {test['test_f1'] - selected['final_val_f1']:+.4f})")
    out(f"accuracy на test p = {test['accuracy']:.4f}, n = {test['n']}, "
        f"SE = sqrt(p(1−p)/n) = {test['se_accuracy']:.4f}")

    out("\nМатриця плутанини (test):\n")
    out(format_confusion(test["confusion"]))
    plots.plot_confusion(test["confusion"], config.FIGURES_DIR / "confusion.png",
                         f"Матриця плутанини на test: {regime}, зерно {seed}")

    out("\nП'ять характерних помилок:\n")
    out(format_errors(test["errors"]))
    plots.plot_errors(test["errors"], config.FIGURES_DIR / "errors.png")

    (config.LAB_DIR / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\nРезультати записано в results.md")


if __name__ == "__main__":
    main()
