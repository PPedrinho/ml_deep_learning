from pathlib import Path


# Caminho raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Diretório para salvar modelos
MODELS_DIR = PROJECT_ROOT / "models"


# Diretório para salvar gráficos e resultados
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Configurações do treinamento
EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2


# Configurações das imagens
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3


# Número de classes do CIFAR-10
NUM_CLASSES = 10


# Nomes das classes
CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


# Cria os diretórios necessários
MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)