from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_DIR = PROJECT_ROOT / "data"

AUTHORIZED_DIR = DATA_DIR / "authorized"

TEST_DIR = DATA_DIR / "test"

RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR = PROJECT_ROOT / "models"


MODEL_NAME = "Facenet512"

DISTANCE_METRIC = "cosine"

THRESHOLD = 0.35

GREEN = (0, 255, 0)

RED = (0, 0, 255)

WHITE = (255, 255, 255)


AUTHORIZED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)