import glob
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import ta
import os

from loguru import logger
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

logger.add("debug.log", level="INFO", rotation="1 MB", backtrace=True, diagnose=True)

DATA_PATH = 'forex_test_data/*.csv'
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../output')

files = glob.glob(DATA_PATH)

if not files:
    logger.error("No CSV files found in the specified directory.")

prices = pd.read_csv(files[0], header=0)

for file in files[1:]:
    df = pd.read_csv(file, header=0)
    prices = pd.merge(prices, df, on='Date', how='inner')

prices['Date'] = pd.to_datetime(prices['Date'])
prices.set_index('Date', inplace=True)

prices_hourly = prices.resample('1h').last().ffill()

def calculate_weights_ffd(degree: float, threshold: float) -> np.ndarray:
    weights = [1.]
    k = 1
    while abs(weights[-1]) >= threshold:
        weights.append(-weights[-1] / k * (degree - k + 1))
        k += 1
    return np.array(weights[::-1])[1:]

def fractional_difference_fixed_single(series: pd.Series, degree: float, threshold: float = 1e-1) -> pd.Series:
    weights = calculate_weights_ffd(degree, threshold)
    width = len(weights)
    series_filtered = series.ffill().dropna()
    values = series_filtered.values
    res = np.convolve(values, weights, mode='valid')
    result_series = pd.Series(index=series_filtered.index[width - 1:], data=res)
    return result_series.reindex(series.index)

def fracdiff_log_price(input_series: pd.Series, threshold=0.01, step=0.01, base_p_value=0.05) -> pd.Series:
    log_price = np.log(input_series)
    degree = -step
    p_value = 1
    while p_value > base_p_value:
        degree += step
        differentiated = fractional_difference_fixed_single(log_price, degree, threshold=threshold)
        p_value = adfuller(differentiated.dropna(), maxlag=1, regression='c', autolag=None)[1]
    return differentiated

def get_volat_w_log_returns(close: pd.Series, span: int = 5) -> pd.Series:
    returns = np.log(close).diff()
    return returns.ewm(span=span).std()

def calculate_variation_of_information(x: np.ndarray, y: np.ndarray, bins: int, norm: bool = False) -> float:
    histogram_xy = np.histogram2d(x, y, bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=histogram_xy)
    marginal_x = ss.entropy(np.histogram(x, bins)[0])
    marginal_y = ss.entropy(np.histogram(y, bins)[0])
    variation_xy = marginal_x + marginal_y - 2 * mutual_information
    if norm:
        joint_xy = marginal_x + marginal_y - mutual_information
        variation_xy /= joint_xy
    return variation_xy

def calculate_number_of_bins(num_observations: int, correlation: float = None, max_bins: int = 1000) -> int:
    try:
        if correlation is None or num_observations <= 1:
            return max_bins
        if correlation == 1:
            return 1
        z = (8 + 324 * num_observations + 12 * (36 * num_observations + 729 * num_observations ** 2) ** .5) ** (1 / 3.)
        bins = round(z / 6. + 2. / (3 * z) + 1. / 3)
        return int(min(bins, max_bins))
    except OverflowError:
        return max_bins

def calculate_variation_of_information_extended(x: np.ndarray, y: np.ndarray, norm: bool = False) -> float:
    num_bins = calculate_number_of_bins(x.shape[0], correlation=np.corrcoef(x, y)[0, 1])
    histogram_xy = np.histogram2d(x, y, num_bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=histogram_xy)
    marginal_x = ss.entropy(np.histogram(x, num_bins)[0])
    marginal_y = ss.entropy(np.histogram(y, num_bins)[0])
    variation_xy = marginal_x + marginal_y - 2 * mutual_information
    if norm:
        joint_xy = marginal_x + marginal_y - mutual_information
        variation_xy /= joint_xy
    return variation_xy

def calculate_mutual_information(x: np.ndarray, y: np.ndarray, norm: bool = False) -> float:
    num_bins = calculate_number_of_bins(x.shape[0], correlation=np.corrcoef(x, y)[0, 1])
    histogram_xy = np.histogram2d(x, y, num_bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=histogram_xy)
    if norm:
        marginal_x = ss.entropy(np.histogram(x, num_bins)[0])
        marginal_y = ss.entropy(np.histogram(y, num_bins)[0])
        mutual_information /= min(marginal_x, marginal_y)
    return mutual_information

def exit_time(close: pd.Series, number_hours: int) -> pd.DataFrame:
    timestamp_array = close.index.searchsorted(close.index + pd.Timedelta(hours=number_hours))
    timestamp_array = timestamp_array[timestamp_array < close.shape[0]]
    timestamp_array = pd.Series(close.index[timestamp_array], index=close.index[:timestamp_array.shape[0]])
    return timestamp_array

for currency in prices_hourly.columns:
    close = prices_hourly[currency]
    features = pd.DataFrame(index=close.index)
    features['frac_diff_log'] = fracdiff_log_price(close)
    features['volat'] = get_volat_w_log_returns(close, span=10)
    features['proc'] = close.pct_change(24) * 100

    weights = np.arange(1, 25)
    features['wpc'] = close.rolling(window=24).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    bb = ta.volatility.BollingerBands(close)
    features['bb_upper'] = bb.bollinger_hband()
    features['bb_middle'] = bb.bollinger_mavg()
    features['bb_lower'] = bb.bollinger_lband()

    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["macd"] = macd_line - signal_line

    features['sma_10'] = ta.trend.SMAIndicator(close, window=10).sma_indicator()
    features['sma_20'] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    features['ema_10'] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    features['ema_20'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()

    for lag in range(1, 6):
        features[f'lag_{lag}'] = close.shift(lag)

    features["dpo"] = close.shift(12) - close.rolling(window=24).mean()
    features['rsi_24'] = ta.momentum.RSIIndicator(close, window=24, fillna=True).rsi()
    features['mom_10'] = ta.momentum.ROCIndicator(close, window=10).roc()
    features['kama_10'] = ta.momentum.KAMAIndicator(close, window=10).kama()
    features['momentum'] = close.diff(10)

    features['med_price'] = close.rolling(window=24).mean()
    features['typ_price'] = (close + close.shift(1) + close.shift(2)) / 3
    weights = np.arange(1, 25)
    features['wcl_price'] = close.rolling(window=24).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    features.dropna(inplace=True)
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), index=features.index)

    selected_features = set(scaled_features.columns)
    for i, col_i in enumerate(scaled_features.columns):
        for col_j in scaled_features.columns[i + 1:]:
            x = scaled_features[col_i].values
            y = scaled_features[col_j].values
            vi = calculate_variation_of_information_extended(x, y, norm=True)
            mi = calculate_mutual_information(x, y, norm=True)
            if vi < 0.2 or mi > 0.8:
                selected_features.discard(col_j)

    final_features = scaled_features[list(selected_features)]

    target = np.sign((close.pct_change()).shift(-1))
    target.fillna(0, inplace=True)

    times = pd.DataFrame(exit_time(close, 24))
    times['end_time'] = times.values
    times['start_time'] = pd.to_datetime(times.index)
    times = times[['end_time', 'start_time']]

    index = final_features.dropna().index.intersection(target.dropna().index).intersection(times.dropna().index)
    features = final_features.loc[index]
    target = target.loc[index]
    target.name = 'target'
    close = close.loc[index]
    times = times.loc[index]
    
    print("Current working directory:", os.getcwd())
    print("Output folder path:", OUTPUT_FOLDER)

    features.to_csv(f"{OUTPUT_FOLDER}/{currency}_features.csv")
    target.to_csv(f"{OUTPUT_FOLDER}/{currency}_target.csv")
    close.to_csv(f"{OUTPUT_FOLDER}/{currency}_close.csv")
    times.to_csv(f"{OUTPUT_FOLDER}/{currency}_times.csv")