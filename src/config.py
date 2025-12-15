from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for dir_path in [PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

DICTIONARY_FILE = RAW_DATA_DIR / "wiki_dictionary.txt"
WORD_FREQ_FILE = RAW_DATA_DIR / "wiki_words_freq10.json"
STATS_FILE = RAW_DATA_DIR / "wiki_stats.json"

DATASET_FILE = PROCESSED_DATA_DIR / "spell_check_dataset.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.10

NUM_SAMPLES = 80000
NUM_CORRECT_SAMPLES = 20000
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 20

ERROR_TYPES = [
    'insertion',
    'deletion',
    'substitution',
    'transposition',
    'correct'
]

USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

print(f"Config loaded - LARGE DATASET MODE")
print(f"  - Device: {DEVICE}")
if USE_GPU:
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  - Target samples: {NUM_SAMPLES + NUM_CORRECT_SAMPLES:,}")