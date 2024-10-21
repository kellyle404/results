import pandas as pd
import h2o
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from h2o.automl import H2OAutoML
from config import (
    LABELS_OUTPUT_DIR,
    ALL_MODELS_OUTPUT_DIR,
    LOGS_DIR,
    MAX_MODELS,
    MAX_RUNTIME_SECS,
    SEED
)

logger.add(
    LOGS_DIR / "02_allmodels_strat1.log",
    level="INFO",
    rotation="1 MB",
    backtrace=True,
    diagnose=True
)

def initialise_h2o():
    logger.info("Starting H2O instance")
    try:
        h2o.init(nthreads=-1, verbose=False)
        h2o.no_progress()
        logger.info("H2O instance started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize H2O: {e}")
        raise

def load_currency_data(input_dir: Path) -> dict:
    logger.info(f"Loading currency data from: {input_dir}")
    feature_files = list(input_dir.glob("*_features.csv"))
    
    if not feature_files:
        logger.error(f"No feature files found in {input_dir}")
        raise FileNotFoundError(f"No feature files found in {input_dir}")

    currency_data = {}

    for file in feature_files:
        currency = file.stem.split('_')[0]
        logger.info(f"Processing data for currency: {currency}")

        target_file = input_dir / f"{currency}_target.csv"
        if not target_file.exists():
            logger.error(f"Target file missing for {currency}")
            continue

        try:
            features = pd.read_csv(file, index_col=0)
            target = pd.read_csv(target_file, index_col=0)

            features_train, features_test, target_train, target_test = train_test_split(
                features, target, train_size=0.7, shuffle=False
            )

            train = pd.concat([features_train, target_train], axis=1)

            currency_data[currency] = {
                'train': train,
                'features': features,
                'target': target,
            }
            logger.info(f"Successfully loaded data for {currency}")
        except Exception as e:
            logger.error(f"Failed to load data for {currency}: {e}")

    return currency_data

def train_and_save_models(currency_data: dict, output_dir: Path):
    """Train H2O models and save them to output directory"""
    total_currencies = len(currency_data)

    for count, (currency, data) in enumerate(currency_data.items(), start=1):
        logger.info(f"Training model {count}/{total_currencies} for {currency}")
        
        train = data["train"]
        logger.info(f"Training on {len(train)} rows for {currency}")
        train_h2o = h2o.H2OFrame(train)

        y = "target"
        x = train_h2o.columns
        x.remove(y)

        train_h2o[y] = train_h2o[y].asfactor()

        try:
            h2o_automl = H2OAutoML(
                max_models=MAX_MODELS,
                max_runtime_secs=MAX_RUNTIME_SECS,
                seed=SEED
            )

            h2o_automl.train(x=x, y=y, training_frame=train_h2o)
            leaderboard = h2o_automl.leaderboard.as_data_frame(use_multi_thread=True)

            logger.info(f"Leaderboard for {currency}:\n{leaderboard}")

            for model_id in leaderboard["model_id"]:
                model = h2o.get_model(model_id)
                h2o.save_model(model, path=str(output_dir), force=True)
                
                old_path = output_dir / model.model_id
                new_path = output_dir / f"{currency}_{model_id}"
                old_path.rename(new_path)

                logger.info(f"Saved model: {currency}_{model_id}")
        except Exception as e:
            logger.error(f"Training failed for {currency}: {e}")

def main():
    try:
        initialize_h2o()
        currency_data = load_currency_data(LABELS_OUTPUT_DIR)
        train_and_save_models(currency_data, ALL_MODELS_OUTPUT_DIR)
        logger.info("Model training and saving completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
    finally:
        h2o.cluster().shutdown()
        logger.info("H2O cluster shut down")

if __name__ == "__main__":
    main()