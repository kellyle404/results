import os
import glob
import pickle
import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import config
from config import (
    LABELS_OUTPUT_DIR, 
    ALL_MODELS_OUTPUT_DIR, 
    BEST_MODELS_OUTPUT_DIR, 
    PREDICTIONS_OUTPUT_DIR, 
    CUT_OFF
)

logger.add(
    LOGS_DIR / "04_predictions_strat1.log",
    level="INFO",
    rotation="1 MB",
    backtrace=True,
    diagnose=True
)

def initialise_h2o():
    try:
        h2o.init(nthreads=-1, verbose=False)
        h2o.no_progress()
    except Exception as e:
        raise RuntimeError(f"Failed to initialise H2O: {e}")

def calc_sharpe_daily(daily_pnl: pd.Series) -> float:
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()
    if std_daily == 0.0:
        return 0.0
    return mean_daily / std_daily

def load_model_rankings() -> dict:
    all_currency_sharpe = {}
    for filename in os.listdir(BEST_MODELS_OUTPUT_DIR):
        currency = filename.split('_')[0]
        file_path = os.path.join(BEST_MODELS_OUTPUT_DIR, filename)
        df = pd.read_csv(file_path)
        all_currency_sharpe[currency] = df
    return all_currency_sharpe

def display_currency_sharpe(all_currency_sharpe: dict):
    for curr, df in all_currency_sharpe.items():
        print(f"\nCurrency: {curr}")
        display(df)

def select_top_models(all_currency_sharpe: dict) -> dict:
    selected_models = {}
    for currency, df in all_currency_sharpe.items():
        selected_models[currency] = {}
        for cv_method, group in df.groupby('cv_method'):
            top_models = group.nlargest(CUT_OFF, 'average_sharpe')
            selected_models[currency][cv_method] = top_models[['model', 'average_sharpe']].reset_index(drop=True)
    return selected_models

def display_selected_models(selected_models: dict):
    for currency, models in selected_models.items():
        print(f"\nCurrency: {currency}")
        for cv_method, model_df in models.items():
            print(f"\n  {cv_method}")
            display(model_df)

def load_currency_data() -> dict:
    feature_files = glob.glob(os.path.join(LABELS_OUTPUT_DIR, "*_features.csv"))
    currency_data = {}
    for file in feature_files:
        currency = os.path.basename(file).split('_')[0]
        features = pd.read_csv(os.path.join(LABELS_OUTPUT_DIR, f"{currency}_features.csv"), index_col=0)
        target = pd.read_csv(os.path.join(LABELS_OUTPUT_DIR, f"{currency}_target.csv"), index_col=0)
        close = pd.read_csv(os.path.join(LABELS_OUTPUT_DIR, f"{currency}_close.csv"), index_col=0)
        times = pd.read_csv(os.path.join(LABELS_OUTPUT_DIR, f"{currency}_times.csv"), index_col=0)
        
        features.index = pd.to_datetime(features.index, format='mixed')
        times.index = pd.to_datetime(times.index)
        close.index = pd.to_datetime(close.index)
        target.index = pd.to_datetime(target.index)
        
        features_train, features_test, target_train, target_test = train_test_split(features, target, train_size=0.7, shuffle=False)
        train = pd.concat([features_train, target_train], axis=1)
        test = pd.concat([features_test, target_test], axis=1)
        
        currency_data[currency] = {'test': test, 'features': features, 'target': target, 'close': close, 'times': times}
    return currency_data

def load_model_paths_with_currency(directory: str, selected_models: dict) -> dict:
    models = {}
    for model_name in os.listdir(directory):
        currency = model_name.split('_')[0]
        model_path = os.path.join(directory, model_name)
        if currency in selected_models:
            for cv_method, model_df in selected_models[currency].items():
                if model_name in model_df['model'].values:
                    if currency not in models:
                        models[currency] = {}
                    if cv_method not in models[currency]:
                        models[currency][cv_method] = []
                    models[currency][cv_method].append(model_path)
    return models

def predict_on_test_set_h2o(test_set, close, model_paths):
    results = {}
    for model_path in model_paths:
        model = h2o.load_model(model_path)
        model_id = model.model_id
        test_set_h2o = h2o.H2OFrame(test_set)
        predictions = model.predict(test_set_h2o).as_data_frame(use_multi_thread=True)
        predicted_class = pd.Series(predictions["predict"].values, index=test_set.index)
        positions = pd.Series(np.where(predicted_class == 1, 1, np.where(predicted_class == -1, -1, 0)), index=test_set.index)
        close_indexed = close.reindex(test_set.index).squeeze()
        strat_returns = (close_indexed.diff() * positions.shift()).fillna(0)
        daily_pnl = strat_returns.resample("D").sum()
        results[model_id] = {"daily_pnl": daily_pnl}
        non_annualised_sharpe = calc_sharpe_daily(daily_pnl)
        print(f"    {model_id} - non annualised sharpe: {non_annualised_sharpe:.4f} - annualised: {non_annualised_sharpe * np.sqrt(252):.4f}")
        results[model_id].update({"sharpe": non_annualised_sharpe})
    return results

def main():
    try:
        initialise_h2o()

        all_currency_sharpe = load_model_rankings()
        display_currency_sharpe(all_currency_sharpe)
        
        selected_models = select_top_models(all_currency_sharpe)
        display_selected_models(selected_models)
        
        currency_data = load_currency_data()

        model_paths = load_model_paths_with_currency(ALL_MODELS_OUTPUT_DIR, selected_models)
        
        predictions_results = {}
        for currency, cv_methods in selected_models.items():
            print(f"\n===========================\n currency: {currency}\n===========================")
            predictions_results[currency] = {}
            for cv_method, model_df in cv_methods.items():
                print(f"  cv method: {cv_method}")
                model_names = set(model_df['model'].values)
                filtered_model_paths = [path for path in model_paths.get(currency, {}).get(cv_method, []) if os.path.basename(path) in model_names]
                test_set = currency_data[currency]['test']
                close_prices = currency_data[currency]['close']
                if filtered_model_paths:
                    results = predict_on_test_set_h2o(test_set, close_prices, filtered_model_paths)
                    predictions_results[currency][cv_method] = results

        output_file_path = os.path.join(PREDICTIONS_OUTPUT_DIRECTORY, 'predictions_results.pkl')
        with open(output_file_path, 'wb') as f:
            pickle.dump(predictions_results, f)

    except Exception as e:
        print(f"Processing failed: {e}")
        raise
    finally:
        h2o.cluster().shutdown()
        print("H2O cluster shut down")

if __name__ == "__main__":
    main()
