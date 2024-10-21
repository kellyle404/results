from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'forex_data'
OUTPUT_DIR = ROOT_DIR / 'output'
LOGS_DIR = ROOT_DIR / 'logs' 

LABELS_OUTPUT_DIR = OUTPUT_DIR / 'labels_strat1'
ALL_MODELS_OUTPUT_DIR = OUTPUT_DIR / 'all_models_strat1_h2o'
BEST_MODELS_OUTPUT_DIR = OUTPUT_DIR / 'best_models_strat1_h2o'
PREDICTIONS_OUTPUT_DIR = OUTPUT_DIR / 'predictions_strat1_h2o'
METRICS_OUTPUT_DIR = OUTPUT_DIR / 'metrics_strat1_h2o'

MAX_MODELS = 50
MAX_RUNTIME_SECS = 60
SEED = 404
S_PARTITIONS = 16  # partitions in CSCV - PBO
CUT_OFF = 45     

for directory in [
    LABELS_OUTPUT_DIR,
    ALL_MODELS_OUTPUT_DIR,
    BEST_MODELS_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
    LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)