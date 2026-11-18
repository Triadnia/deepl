import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LAB_DIR = Path(__file__).resolve().parent
SPLIT_CSV = LAB_DIR / "split.csv"
RUNS_DIR = LAB_DIR / "runs"
FIGURES_DIR = LAB_DIR / "figures"
MODEL_PATH = LAB_DIR / "lab3_model.pt"

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASSES)
TRAIN_DIR = "seg_train"
TEST_DIR = "seg_test"
SPLIT_SEED = 0
TRAIN_PER_CLASS = 200
PARTS = ["train", "validation", "test"]

SPLIT_CONTROL = {
    ("buildings", "train"): (200, 1_996_761),
    ("buildings", "validation"): (218, 4_837_323),
    ("buildings", "test"): (219, 4_837_559),
    ("forest", "train"): (200, 2_032_625),
    ("forest", "validation"): (237, 5_276_962),
    ("forest", "test"): (237, 5_291_326),
    ("glacier", "train"): (200, 2_183_923),
    ("glacier", "validation"): (276, 6_135_652),
    ("glacier", "test"): (277, 6_157_250),
    ("mountain", "train"): (200, 1_950_345),
    ("mountain", "validation"): (262, 5_789_535),
    ("mountain", "test"): (263, 5_849_765),
    ("sea", "train"): (200, 1_987_613),
    ("sea", "validation"): (255, 5_655_068),
    ("sea", "test"): (255, 5_668_011),
    ("street", "train"): (200, 1_841_817),
    ("street", "validation"): (250, 5_541_845),
    ("street", "test"): (251, 5_547_533),
}

IMG_SIZE = 160
CROP_SCALE = (0.7, 1.0)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EPOCHS = 8
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
NUM_WORKERS = 0
STEPS_PER_EPOCH = 38
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
WEIGHT_DECAY = 1e-4

TOTAL_PARAMS = 11_179_590
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

REGIMES = {
    "scratch": {
        "pretrained": False,
        "trainable": ["*"],
        "eval_modules": [],
        "groups": [(["*"], LR_HEAD)],
        "trainable_params": 11_179_590,
    },
    "frozen": {
        "pretrained": True,
        "trainable": ["fc"],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
        "groups": [(["fc"], LR_HEAD)],
        "trainable_params": 3_078,
    },
    "unfreeze": {
        "pretrained": True,
        "trainable": ["layer4", "fc"],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "groups": [(["fc"], LR_HEAD), (["layer4"], LR_BACKBONE)],
        "trainable_params": 8_396_806,
    },
}

RUNS = [
    ("scratch", 1),
    ("frozen", 1), ("frozen", 2), ("frozen", 3),
    ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3),
]

CHECK_SEED = 1
CHECK_STEPS = 5
