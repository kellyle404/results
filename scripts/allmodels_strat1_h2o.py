import pandas as pd
import h2o
import os
import glob
from loguru import logger
from sklearn.model_selection import train_test_split
from h2o.automl import H2OAutoML

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(ROOT_DIR, 'output', 'labels_strat1')
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', 'all_models_strat1_h2o')

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

logger.add(
    os.path.join(MODEL_OUTPUT_DIR, "model_training.log"),
    level="INFO",
    rotation="1 MB",
    backtrace=True,
    diagnose=True
)

logger.info("Starting H2O instance")
h2o.init(max_mem_size="2G", nthreads=-1, verbose=False)
h2o.no_progress()

def load_currency_data(input_dir):
    logger.info(f"Loading currency data from: {input_dir}")
    feature_files = glob.glob(os.path.join(input_dir, "*_features.csv"))
    
    if not feature_files:
        logger.error(f"No feature files found in {input_dir}")
        raise FileNotFoundError(f"No feature files found in {input_dir}")

    currency_data = {}

    for file in feature_files:
        currency = os.path.basename(file).split('_')[0]
        logger.info(f"Processing data for currency: {currency}")

        try:
            features = pd.read_csv(file, index_col=0)
            target = pd.read_csv(os.path.join(input_dir, f"{currency}_target.csv"), index_col=0)

            features_train, features_test, target_train, target_test = train_test_split(
                features, target, train_size=0.7, shuffle=False
            )

            train = pd.concat([features_train, target_train], axis=1)

            currency_data[currency] = {
                'train': train,
                'features': features,
                'target': target,
            }
        except Exception as e:
            logger.error(f"Failed to load data for {currency}: {e}")

    return currency_data

def train_and_save_models(currency_data, output_dir):
    total_currencies = len(currency_data)

    for count, (currency, data) in enumerate(currency_data.items(), start=1):
        logger.info(f"Training model {count}/{total_currencies} for {currency}")
        
        train = data["train"][:5000]
        train_h2o = h2o.H2OFrame(train)

        y = "target"
        x = train_h2o.columns
        x.remove(y)

        train_h2o[y] = train_h2o[y].asfactor()

        try:
            h2o_automl = H2OAutoML(
                max_models=2,
                seed=404,
                max_runtime_secs=30,
                excluded_models=["StackedEnsemble"]
            )

            h2o_automl.train(x=x, y=y, training_frame=train_h2o)
            leaderboard = h2o_automl.leaderboard.as_data_frame(use_multi_thread=True)

            for model_id in leaderboard["model_id"]:
                model = h2o.get_model(model_id)
                model_name_with_currency = f"{currency}_{model_id}"

                h2o.save_model(
                    model, 
                    path=output_dir, 
                    force=True, 
                    filename=model_name_with_currency
                )

                logger.info(f"Saved model: {model_name_with_currency}")
        except Exception as e:
            logger.error(f"Training failed for {currency}: {e}")

def main():
    try:
        currency_data = load_currency_data(INPUT_DIR)
        train_and_save_models(currency_data, MODEL_OUTPUT_DIR)
        logger.info("Model training and saving completed successfully")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
