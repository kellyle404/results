import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools as itt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from loguru import logger
from config import PREDICTIONS_OUTPUT_DIR, METRICS_OUTPUT_DIR, LOGS_DIR

sns.set_style('dark')

logger.add(
    LOGS_DIR / "05_metrics_strat1.log",
    level="INFO",
    rotation="1 MB",
    backtrace=True,
    diagnose=True
)

def calc_sharpe_daily(pnl: pd.Series) -> float:
    daily_pnl = pnl.resample("D").sum()
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std()
    if std_daily == 0.0:
        logger.warning("Standard deviation of daily PnL is zero, returning 0.0 for Sharpe ratio.")
        return 0.0
    return mean_daily / std_daily

def benchmark_sharpe(sharpe_ratio_estimates: list) -> float:
    standard_deviation = np.array(sharpe_ratio_estimates).std()
    benchmark_value = standard_deviation * ((1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / len(sharpe_ratio_estimates)) + np.euler_gamma * ss.norm.ppf(1 - 1 / len(sharpe_ratio_estimates) * np.e ** (-1)))
    return benchmark_value

def deflated_sharpe(observed_sharpe_ratio: float, benchmark_sharpe_ratio: float, number_of_returns: int, skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    numerator = (observed_sharpe_ratio - benchmark_sharpe_ratio) * np.sqrt(number_of_returns - 1)
    denominator = (1 - skewness_of_returns * observed_sharpe_ratio + (kurtosis_of_returns - 1) / 4 * observed_sharpe_ratio**2) ** 0.5
    test_value = numerator / denominator
    return ss.norm.cdf(test_value)

def calc_deflated_sharpe(results_test_set):
    all_sharpes = [model_results["sharpe"] for model_results in results_test_set.values()]
    benchmark = benchmark_sharpe(all_sharpes)
    highest_sharpe = max(all_sharpes)
    highest_sharpe_model = None
    strat_returns = None
    for model_id, model_results in results_test_set.items():
        if model_results["sharpe"] == highest_sharpe:
            highest_sharpe_model = model_id
            strat_returns = model_results["daily_pnl"]
    number_of_returns = len(strat_returns)
    skewness_of_returns = strat_returns.skew()
    kurtosis_of_returns = strat_returns.kurtosis()
    dsr = deflated_sharpe(observed_sharpe_ratio=highest_sharpe, benchmark_sharpe_ratio=benchmark, number_of_returns=number_of_returns, skewness_of_returns=skewness_of_returns, kurtosis_of_returns=kurtosis_of_returns)
    logger.info(f"Calculated DSR: {dsr} for model: {highest_sharpe_model} with Sharpe: {highest_sharpe}")
    return dsr

def calc_pbo(model_data, S):
    M = pd.DataFrame({model: data['daily_pnl'] for model, data in model_data.items()})
    submatrices = [M.iloc[i::S, :] for i in range(S)]
    combinations_size = S // 2
    combinations_indices = list(itt.combinations(range(S), combinations_size))
    logits = []
    for comb in combinations_indices:
        J_train = pd.concat([submatrices[i] for i in comb], axis=0)
        J_test = pd.concat([submatrices[i] for i in range(S) if i not in comb], axis=0)
        Rc_IS = J_train.mean()
        Rc_OOS = J_test.mean()
        n_star = Rc_IS.idxmax()
        rank_OOS = Rc_OOS.rank(ascending=False)[n_star]
        omega_c = rank_OOS / (len(M) + 1)
        logit_lambda_c = np.log(omega_c / (1 - omega_c))
        logits.append(logit_lambda_c)
    failures = sum(1 for logit in logits if logit <= 0)
    pbo = failures / len(logits)
    logger.info(f"Calculated PBO: {pbo}")
    return pbo

def dict_to_df(results_dict, metric_name):
    df_list = []
    for currency, methods in results_dict.items():
        for method, value in methods.items():
            df_list.append({'currency': currency, 'cv method': method, 'value': value, 'metric': metric_name})
    return pd.DataFrame(df_list)

def plot_metric(df, metric_name):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='cv method', y='value', data=df, palette='rocket', hue='cv method', width=0.2)
    plt.title(f"\n{metric_name}")
    plt.ylabel(metric_name)
    plt.xlabel('')
    plt.tight_layout()

def main():
    try:
        input_file_path = os.path.join(PREDICTIONS_OUTPUT_DIR, 'predictions_results.pkl')
        with open(input_file_path, 'rb') as f:
            predictions_results = pickle.load(f)

        logger.info("Loaded predictions results successfully.")

        dsr_results = {}
        for currency, cv_methods in predictions_results.items():
            dsr_results[currency] = {}
            for cv_method, models in cv_methods.items():
                dsr = calc_deflated_sharpe(models)
                dsr_results[currency][cv_method] = dsr

        dsr_output_path = os.path.join(METRICS_OUTPUT_DIR, 'dsr_results.pkl')
        with open(dsr_output_path, 'wb') as f:
            pickle.dump(dsr_results, f)

        logger.info("DSR results saved successfully.")

        pbo_results = {}
        S = 10  # Define S based on your requirements
        for currency, cv_methods in predictions_results.items():
            pbo_results[currency] = {}
            for cv_method, model_data in cv_methods.items():
                pbo_value = calc_pbo(model_data, S)
                pbo_results[currency][cv_method] = pbo_value

        pbo_output_path = os.path.join(METRICS_OUTPUT_DIR, 'pbo_results.pkl')
        with open(pbo_output_path, 'wb') as f:
            pickle.dump(pbo_results, f)

        logger.info("PBO results saved successfully.")

        dsr_df = dict_to_df(dsr_results, 'DSR')
        pbo_df = dict_to_df(pbo_results, 'PBO')
        df = pd.concat([dsr_df, pbo_df], ignore_index=True)

        metrics_to_plot = [(dsr_df, 'DSR'), (pbo_df, 'PBO')]
        for df, metric in metrics_to_plot:
            plot_metric(df, metric)
            logger.info(f"Plotted metric: {metric}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
