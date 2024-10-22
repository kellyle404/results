import os

ROOT_DIR = os.path.abspath(os.path.join(__file__, "../.."))
DATA_DIR = os.path.join(ROOT_DIR, 'forex_data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

LABELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'labels_strat1')
ALL_MODELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'all_models_strat1_h2o')
BEST_MODELS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'best_models_strat1_h2o')
PREDICTIONS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'predictions_strat1_h2o')
METRICS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'metrics_strat1_h2o')

MAX_MODELS = 3
MAX_RUNTIME_SECS = 10
SEED = 404
S_PARTITIONS = 4  
CUT_OFF = 2     

for directory in [
    LABELS_OUTPUT_DIR,
    ALL_MODELS_OUTPUT_DIR,
    BEST_MODELS_OUTPUT_DIR,
    PREDICTIONS_OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
    LOGS_DIR
]:
    os.makedirs(directory, exist_ok=True)
