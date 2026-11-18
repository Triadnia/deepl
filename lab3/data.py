import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def part_dir(data_root: Path, name: str) -> Path:
    nested = data_root / name / name
    return nested if nested.is_dir() else data_root / name


def list_class_files(class_dir: Path) -> list[str]:
    return sorted(p.name for p in class_dir.iterdir() if p.suffix.lower() == ".jpg")


def build_split(data_root: Path) -> list[dict]:
    rows = []

    rng = np.random.default_rng(config.SPLIT_SEED)
    for label in config.CLASSES:
        files = list_class_files(part_dir(data_root, config.TRAIN_DIR) / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        rows += [{"file": files[i], "label": label, "part": "train"}
                 for i in idx[:config.TRAIN_PER_CLASS]]

    rng = np.random.default_rng(config.SPLIT_SEED)
    for label in config.CLASSES:
        files = list_class_files(part_dir(data_root, config.TEST_DIR) / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        rows += [{"file": files[i], "label": label, "part": "validation"} for i in idx[:cut]]
        rows += [{"file": files[i], "label": label, "part": "test"} for i in idx[cut:]]

    return rows


def write_split(rows: list[dict], path: Path = config.SPLIT_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_split(path: Path = config.SPLIT_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def control_table(rows: list[dict]) -> dict[tuple[str, str], tuple[int, int]]:
    table = {key: (0, 0) for key in config.SPLIT_CONTROL}
    for row in rows:
        key = (row["label"], row["part"])
        count, total = table[key]
        table[key] = (count + 1, total + int(Path(row["file"]).stem))
    return table


def find_overlaps(rows: list[dict]) -> list[str]:
    counts = Counter(row["file"] for row in rows)
    return sorted(name for name, n in counts.items() if n > 1)


def format_control_table(table: dict) -> tuple[str, bool]:
    lines = [f"{'Клас':<10}" + "".join(f"{part:>26}" for part in config.PARTS)]
    all_match = True
    for label in config.CLASSES:
        cells = []
        for part in config.PARTS:
            got = table[(label, part)]
            ok = got == config.SPLIT_CONTROL[(label, part)]
            all_match &= ok
            cells.append(f"{got[0]} / {got[1]:,}".replace(",", " ") + (" ✓" if ok else " ✗"))
        lines.append(f"{label:<10}" + "".join(f"{c:>26}" for c in cells))
    totals = []
    for part in config.PARTS:
        count = sum(table[(label, part)][0] for label in config.CLASSES)
        total = sum(table[(label, part)][1] for label in config.CLASSES)
        totals.append(f"{count} / {total:,}".replace(",", " ") + "  ")
    lines.append(f"{'Разом':<10}" + "".join(f"{c:>26}" for c in totals))
    return "\n".join(lines), all_match


def check_split(rows: list[dict]) -> bool:
    text, all_match = format_control_table(control_table(rows))
    overlaps = find_overlaps(rows)
    print(text)
    print(f"Усі 18 клітинок збігаються з контрольною таблицею: {'так' if all_match else 'НІ'}")
    print(f"Файлів, що входять у дві частини: {len(overlaps)}")
    return all_match and not overlaps


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMG_SIZE, scale=config.CROP_SCALE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])


class SplitDataset(Dataset):
    def __init__(self, data_root: Path, part: str, transform):
        source = config.TRAIN_DIR if part == "train" else config.TEST_DIR
        root = part_dir(data_root, source)
        rows = [row for row in read_split() if row["part"] == part]
        self.files = [row["file"] for row in rows]
        self.paths = [root / row["label"] / row["file"] for row in rows]
        self.labels = [config.CLASSES.index(row["label"]) for row in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def make_train_loader(data_root: Path) -> DataLoader:
    dataset = SplitDataset(data_root, "train", train_transform)
    return DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True,
                      drop_last=False, num_workers=config.NUM_WORKERS)


def make_val_loader(data_root: Path) -> DataLoader:
    dataset = SplitDataset(data_root, "validation", eval_transform)
    return DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS)


def make_test_loader(data_root: Path) -> DataLoader:
    dataset = SplitDataset(data_root, "test", eval_transform)
    return DataLoader(dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(config.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.STD).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Поділ на train / validation / test і контроль поділу")
    parser.add_argument("--data", type=Path, required=True,
                        help="тека з розпакованим набором (містить seg_train/ і seg_test/)")
    args = parser.parse_args()

    rows = build_split(args.data)
    write_split(rows)
    print(f"Записано {config.SPLIT_CSV.name}: {len(rows)} рядків")
    if not check_split(read_split()):
        raise SystemExit("Поділ не збігається з контрольною таблицею")


if __name__ == "__main__":
    main()
