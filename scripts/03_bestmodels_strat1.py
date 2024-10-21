import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import h2o
from sklearn.model_selection import train_test_split
import itertools as itt
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import config
from config import (
    DATA_DIR,
    ALL_MODELS_OUTPUT_DIR,
    BEST_MODELS_OUTPUT_DIR,
    LABELS_OUTPUT_DIR,
    LOGS_DIR,
    SEED
)

logger.add(
    LOGS_DIR / "03_bestmodels_strat1.log",
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

def calc_pnl(close: pd.Series, positions: pd.Series) -> pd.Series:
    """Calculate PnL from close prices and positions"""
    return ((close).diff() * positions.shift()).fillna(0)

def calc_sharpe(pnl: pd.Series) -> float:
    """Calculate annualized Sharpe ratio"""
    pnl.fillna(0)
    annualised_pnl = pnl.resample("D").sum()
    return (annualised_pnl.mean() / annualised_pnl.std()) * np.sqrt(252)



class CrossVal(ABC):
    @abstractmethod
    def create_splits(self, data: pd.DataFrame, times: pd.DataFrame, n_groups: int, k_test_groups: int = 1) -> list:
        pass

class CombPurgeEmbargoKFold(CrossVal):
    def create_splits(self, data: pd.DataFrame, times: pd.DataFrame, n_groups: int = 5, k_test_groups: int = 2) -> list:
        indices = np.arange(len(data))
        split_segments = np.array_split(indices, n_groups)
        combinations_ = list(itt.combinations(range(n_groups), k_test_groups))
        splits = []
        for test_groups in combinations_:
            test_indices = {f"group_{i + 1}": split_segments[i] for i in range(n_groups)}
            train_indices = np.setdiff1d(indices, np.concatenate([test_indices[f"group_{g + 1}"] for g in test_groups]))
            times_train = times.iloc[train_indices]
            times_test = times.iloc[np.concatenate([test_indices[f"group_{g + 1}"] for g in test_groups])]
            splits.append({
                "train_indices": train_indices,
                "test_indices": {f"group_{g + 1}": test_indices[f"group_{g + 1}"] for g in test_groups},
                "times_train": times_train[["start_time", "end_time"]],
                "times_test": times_test[["start_time", "end_time"]],
            })
        return splits

class KFold(CrossVal):
    def create_splits(self, data: pd.DataFrame, times: pd.DataFrame, n_groups: int = 5, k_test_groups: int = 2) -> list:
        indices = np.arange(len(data))
        split_segments = np.array_split(indices, n_groups)
        combinations_ = list(itt.combinations(range(n_groups), 1))
        splits = []
        for test_groups in combinations_:
            test_indices = {f"group_{i + 1}": split_segments[i] for i in range(n_groups)}
            train_indices = np.setdiff1d(indices, np.concatenate([test_indices[f"group_{g + 1}"] for g in test_groups]))
            times_train = times.iloc[train_indices]
            times_test = times.iloc[np.concatenate([test_indices[f"group_{g + 1}"] for g in test_groups])]
            splits.append({
                "train_indices": train_indices,
                "test_indices": {f"group_{g + 1}": test_indices[f"group_{g + 1}"] for g in test_groups},
                "times_train": times_train[["start_time", "end_time"]],
                "times_test": times_test[["start_time", "end_time"]],
            })
        return splits

class PurgedKFold(CrossVal):
    def create_splits(self, data: pd.DataFrame, times: pd.DataFrame, n_groups: int = 5) -> list:
        indices = np.arange(len(data))
        split_segments = np.array_split(indices, n_groups)
        combinations_ = list(itt.combinations(range(n_groups), 1))
        splits = []
        for test_groups in combinations_:
            test_indices = {f"group_{i + 1}": split_segments[i] for i in range(n_groups)}
            train_indices = np.setdiff1d(indices, np.concatenate([test_indices[f"group_{g + 1}"] for g in test_groups]))
            times_train = times.iloc[train_indices]
            times_test = times.iloc[np.concatenate([test_indices[f"group_{g + 1}"] for g in test_groups])]
            splits.append({
                "train_indices": train_indices,
                "test_indices": {f"group_{g + 1}": test_indices[f"group_{g + 1}"] for g in test_groups},
                "times_train": times_train[["start_time", "end_time"]],
                "times_test": times_test[["start_time", "end_time"]],
            })
        return splits

class RollingWindow(CrossVal):
    def create_splits(self, data: pd.DataFrame, times: pd.DataFrame, train_size: int, step_size: int) -> list:
        indices = np.arange(len(data))
        splits = []
        start = 0
        group_counter = 1
        while start + train_size + step_size <= len(data):
            train_indices = indices[start: start + train_size]
            test_indices = indices[start + train_size: start + train_size + step_size]
            times_train = times.iloc[train_indices]
            times_test = times.iloc[test_indices]
            splits.append({
                "train_indices": train_indices,
                "test_indices": {f"group_{group_counter}": test_indices},
                "times_train": times_train[["start_time", "end_time"]],
                "times_test": times_test[["start_time", "end_time"]],
            })
            start += step_size
            group_counter += 1
        return splits

class ExpandingWindow(CrossVal):
    def create_splits(self, data: pd.DataFrame, times: pd.DataFrame, initial_train_size: int, step_size: int) -> list:
        indices = np.arange(len(data))
        splits = []
        train_end = initial_train_size
        group_counter = 1
        while train_end + step_size <= len(data):
            train_indices = indices[:train_end]
            test_indices = indices[train_end: train_end + step_size]
            times_train = times.iloc[train_indices]
            times_test = times.iloc[test_indices]
            splits.append({
                "train_indices": train_indices,
                "test_indices": {f"group_{group_counter}": test_indices},
                "times_train": times_train[["start_time", "end_time"]],
                "times_test": times_test[["start_time", "end_time"]],
            })
            train_end += step_size
            group_counter += 1
        return splits

def form_backtest_paths(predictions):
    paths = {}
    for model_id, model_predictions in predictions.items():
        model_paths = {}
        instance_counters = {}
        group_names = set()
        for split_preds in model_predictions.values():
            group_names.update(split_preds.keys())
        instance_counters = {group: 0 for group in group_names}
        for split, split_preds in model_predictions.items():
            for group_name, preds in split_preds.items():
                current_instance = instance_counters[group_name]
                path_key = f"path_{current_instance + 1}"
                if path_key not in model_paths:
                    model_paths[path_key] = {}
                model_paths[path_key][group_name] = {"predictions": preds["predictions"], "dates": preds["date"]}
                instance_counters[group_name] += 1
        paths[model_id] = model_paths
    return paths

def apply_embargo(splits, total_data_size, pct_embargo=0.1):
    embargo_size = int(total_data_size * pct_embargo)
    def get_continuous_ranges(arr):
        ranges = []
        start = arr[0]
        for i in range(1, len(arr)):
            if arr[i] != arr[i - 1] + 1:
                ranges.append((start, arr[i - 1]))
                start = arr[i]
        ranges.append((start, arr[-1]))
        return ranges
    def apply_embargo_to_range(train_range, last_test_index):
        if train_range[0] == last_test_index + 1:
            new_start = min(train_range[0] + embargo_size, train_range[1] + 1)
            return (new_start, train_range[1])
        return train_range
    embargoed_splits = []
    for split in splits:
        new_train_ranges = []
        train_ranges = get_continuous_ranges(np.array(split["train_indices"]))
        last_test_indices = [max(indices) for indices in split["test_indices"].values()]
        for train_range in train_ranges:
            embargoed_range = train_range
            for last_test_index in last_test_indices:
                embargoed_range = apply_embargo_to_range(embargoed_range, last_test_index)
            new_train_ranges.append(embargoed_range)
        purged_train_indices = []
        for start, end in new_train_ranges:
            purged_train_indices.append(np.arange(start, end + 1))
        new_train_indices = np.concatenate(purged_train_indices)
        indices_mask = np.isin(split["train_indices"], new_train_indices)
        new_times_train = split["times_train"].iloc[indices_mask].copy()
        embargoed_splits.append({
            "train_indices": new_train_indices,
            "test_indices": split["test_indices"],
            "times_train": new_times_train,
            "times_test": split["times_test"],
        })
    return embargoed_splits

def apply_purging(splits):
    purged_splits = []
    for split in splits:
        train_indices = split["train_indices"]
        train_times = split["times_train"]
        test_times = split["times_test"]
        train_start = train_times['start_time'].values
        train_end = train_times['end_time'].values
        keep_train_mask = np.ones(len(train_indices), dtype=bool)
        for test_idx in range(len(test_times)):
            test_start = test_times['start_time'].iloc[test_idx]
            test_end = test_times['end_time'].iloc[test_idx]
            overlap = (train_start < test_end) & (train_end > test_start)
            keep_train_mask &= ~overlap
        purged_train_indices = train_indices[keep_train_mask]
        purged_times_train = train_times.iloc[keep_train_mask]
        purged_splits.append({
            "train_indices": purged_train_indices,
            "test_indices": split["test_indices"],
            "times_train": purged_times_train,
            "times_test": test_times,
        })
    return purged_splits

def final_plot_splits(splits):
    plt.figure(figsize=(16, 2))
    sns.set_style('darkgrid')
    for i, split in enumerate(splits):
        plt.scatter(split['train_indices'], [i] * len(split['train_indices']), color='lightgrey', s=5, label='Train' if i == 0 else "")
        test_indices = np.concatenate([indices for indices in split['test_indices'].values()])
        plt.scatter(test_indices, [i] * len(test_indices), color='red', s=5, label='Test' if i == 0 else "")
    plt.yticks(range(len(splits)), [f'Split {i+1}' for i in range(len(splits))])
    plt.xlabel('Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def identify_best_models_h2o(cv_class: CrossVal, data: pd.DataFrame, times: pd.DataFrame, pct_embargo: float = 0.01, n_groups: int = 5, k_test_groups: int = 2, train_size: int = None, step_size: int = None, initial_train_size: int = None) -> pd.DataFrame:
    if isinstance(cv_class, CombPurgeEmbargoKFold):
        splits = cv_class.create_splits(data, times, n_groups=n_groups, k_test_groups=k_test_groups)
        embargoed_splits = apply_embargo(splits, total_data_size = len(data), pct_embargo=pct_embargo)
        purged_splits = apply_purging(embargoed_splits)
    elif isinstance(cv_class, PurgedKFold):
        splits = cv_class.create_splits(data, times, n_groups=n_groups)
        embargoed_splits = apply_embargo(splits, total_data_size = len(data), pct_embargo=pct_embargo)
        purged_splits = apply_purging(embargoed_splits)
    elif isinstance(cv_class, KFold):
        splits = cv_class.create_splits(data, times, n_groups=n_groups, k_test_groups=k_test_groups)
        purged_splits = splits
    elif isinstance(cv_class, RollingWindow):
        splits = cv_class.create_splits(data, times, train_size=train_size, step_size=step_size)
        purged_splits = apply_purging(splits)
    elif isinstance(cv_class, ExpandingWindow):
        splits = cv_class.create_splits(data, times, initial_train_size=initial_train_size, step_size=step_size)
        purged_splits = apply_purging(splits)
    predictions = {}
    for model in loaded_models[currency]:
        model_id = model.model_id
        model_predictions = {}
        for i, split in enumerate(purged_splits):
            train_indices = split["train_indices"]
            test_indices_dict = split["test_indices"]
            train_data = data.iloc[train_indices]
            train_h2o = h2o.H2OFrame(train_data)
            y = "target"
            x = train_h2o.columns
            x.remove(y)
            train_h2o[y] = train_h2o[y].asfactor()
            model.train(x=x, y=y, training_frame=train_h2o)
            split_preds = {}
            for group_name, test_indices in test_indices_dict.items():
                test_data = data.iloc[test_indices]
                test_h2o = h2o.H2OFrame(test_data)
                test_h2o[y] = test_h2o[y].asfactor()
                predictions_h2o = model.predict(test_h2o)
                pred_df = predictions_h2o.as_data_frame(use_multi_thread=True)
                start_times = times.iloc[test_indices]["start_time"].values
                split_preds[group_name] = {"predictions": pred_df, "date": start_times}
            model_predictions[f"split_{i + 1}"] = split_preds
        predictions[model_id] = model_predictions
    all_paths = form_backtest_paths(predictions)
    concatenated_paths = {}
    for model_key, paths in all_paths.items():
        concatenated_paths[model_key] = {}
        for path_key, groups in paths.items():
            concatenated_predictions = np.concatenate([groups[group_name]["predictions"].values for group_name in groups])
            concatenated_date = np.concatenate([groups[group_name]["date"] for group_name in groups])
            concatenated_paths[model_key][path_key] = {
                "predictions": concatenated_predictions,
                "date": concatenated_date,
            }
    sharpe_ratios = {}
    for model_key, paths in concatenated_paths.items():
        sharpe_ratios[model_key] = {}
        for path_key, data in paths.items():
            predictions = data["predictions"][:, 0]
            date = data["date"]
            positions = np.where(predictions == 1, 1, np.where(predictions == -1, -1, 0))
            positions_series = pd.Series(positions, index=pd.to_datetime(date))
            filtered_close = close[close.index.isin(positions_series.index)].squeeze()
            strat_returns = calc_pnl(filtered_close, positions_series[filtered_close.index])
            sharpe_ratio = calc_sharpe(strat_returns.dropna())
            sharpe_ratios[model_key][path_key] = sharpe_ratio
    all_models_cpcv = []
    for model_key, paths in sharpe_ratios.items():
        row = {"model": model_key}
        for path_key, sharpe_value in paths.items():
            row[path_key] = sharpe_value
        average_sharpe = sum(paths.values()) / len(paths) if paths else 0
        row["average_sharpe"] = average_sharpe
        all_models_cpcv.append(row)
    sharpe_df = pd.DataFrame(all_models_cpcv)
    return sharpe_df


def load_currency_data() -> dict:
    logger.info(f"Loading currency data from: {LABELS_OUTPUT_DIR}")
    all_currency_files = list(LABELS_OUTPUT_DIR.glob("*_features.csv"))
    
    if not all_currency_files:
        logger.error(f"No files found in {LABELS_OUTPUT_DIR}")
        raise FileNotFoundError(f"No files found in {LABELS_OUTPUT_DIR}")

    currency_data = {}
    for file in all_currency_files:
        currency = file.stem.split('_')[0]
        logger.info(f"Processing data for currency: {currency}")
        
        try:
            features = pd.read_csv(LABELS_OUTPUT_DIR / f"{currency}_features.csv", index_col=0)
            target = pd.read_csv(LABELS_OUTPUT_DIR / f"{currency}_target.csv", index_col=0)
            close = pd.read_csv(LABELS_OUTPUT_DIR / f"{currency}_close.csv", index_col=0)
            times = pd.read_csv(LABELS_OUTPUT_DIR / f"{currency}_times.csv", index_col=0)

            for df in [features, times, close, target]:
                df.index = pd.to_datetime(df.index)

            features_train, features_test, target_train, target_test = train_test_split(
                features, target, train_size=0.7, shuffle=False
            )
            train = pd.concat([features_train, target_train], axis=1)
            
            currency_data[currency] = {
                'train': train,
                'features': features,
                'target': target,
                'close': close,
                'times': times
            }
            logger.info(f"Successfully loaded data for {currency}")
        except Exception as e:
            logger.error(f"Failed to load data for {currency}: {e}")
            continue

    return currency_data

def load_models() -> dict:
    logger.info(f"Loading models from: {ALL_MODELS_OUTPUT_DIR}")
    models = {}
    for model_path in ALL_MODELS_OUTPUT_DIR.iterdir():
        currency = model_path.stem.split('_')[0]
        if currency not in models:
            models[currency] = []
        models[currency].append(str(model_path))
    return models

def main():
    try:
        initialise_h2o()
        currency_data = load_currency_data()
        loaded_models = load_models()
        
        all_results = {}
        for currency, data_dict in currency_data.items():
            logger.info(f"Processing currency: {currency}")
            
            train = data_dict['train']
            times = data_dict['times']
            close = data_dict['close']
            models_for_currency = loaded_models.get(currency, [])
            
            if not models_for_currency:
                logger.warning(f"No models found for currency: {currency}")
                continue

            cv_methods = {
                "KFold": KFold(),
                "CombPurgeEmbargoKFold": CombPurgeEmbargoKFold(),
                "PurgedKFold": PurgedKFold(),
                "RollingWindow": RollingWindow(),
                "ExpandingWindow": ExpandingWindow(),
            }

            all_results[currency] = {}
            for cv_name, cv_method in cv_methods.items():
                logger.info(f"Running {cv_name} for {currency}")
                try:
                    result_df = identify_best_models_h2o(
                        cv_class=cv_method,
                        data=train,
                        times=times,
                        pct_embargo=0.01,
                        n_groups=5,
                        k_test_groups=2,
                        loaded_models={currency: models_for_currency}
                    )
                    all_results[currency][cv_name] = result_df
                except Exception as e:
                    logger.error(f"Error in {cv_name} for {currency}: {e}")
                    continue

        logger.info("Saving results")
        for currency, cv_results in all_results.items():
            avg_sharpe_df = pd.DataFrame()
            for cv_name, result_df in cv_results.items():
                avg_sharpe = result_df[['model', 'average_sharpe']].copy()
                avg_sharpe['cv_method'] = cv_name
                avg_sharpe_df = pd.concat([avg_sharpe_df, avg_sharpe], ignore_index=True)
            
            output_path = BEST_MODELS_OUTPUT_DIR / f"{currency}_average_sharpe.csv"
            avg_sharpe_df.to_csv(output_path, index=False)
            logger.info(f"Saved results for {currency} to {output_path}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()