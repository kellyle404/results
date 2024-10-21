import glob
import numpy as np
import pandas as pd
import scipy.stats as ss
import ta
from pathlib import Path
from loguru import logger
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import config
from config import (
    DATA_DIR,
    LABELS_OUTPUT_DIR,
    LOGS_DIR
)

logger.add(
    LOGS_DIR / "01_labels_strat1.log",  
    level="INFO",
    rotation="1 MB",
    backtrace=True,
    diagnose=True
)


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
    res = np.convolve(series_filtered.values, weights, mode='valid')
    return pd.Series(data=res, index=series_filtered.index[width - 1:]).reindex(series.index)

def fracdiff_log_price(input_series: pd.Series, threshold=0.01, step=0.01, base_p_value=0.05) -> pd.Series:
    log_price = np.log(input_series)
    degree = -step
    p_value = 1
    while p_value > base_p_value:
        degree += step
        differentiated = fractional_difference_fixed_single(log_price, degree, threshold)
        p_value = adfuller(differentiated.dropna(), maxlag=1, regression='c', autolag=None)[1]
    return differentiated

def get_volat_w_log_returns(close: pd.Series, span: int = 5) -> pd.Series:
    returns = np.log(close).diff()
    return returns.ewm(span=span).std()

def calculate_variation_of_information(x: np.ndarray, y: np.ndarray, bins: int, norm: bool = False) -> float:
    histogram_xy = np.histogram2d(x, y, bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=histogram_xy)
    marginal_x = ss.entropy(np.histogram(x, bins)[0])
    marginal_y = ss.entropy(np.histogram(y, bins)[0])
    variation_xy = marginal_x + marginal_y - 2 * mutual_info
    if norm:
        joint_xy = marginal_x + marginal_y - mutual_info
        variation_xy /= joint_xy
    return variation_xy

def calculate_number_of_bins(num_observations: int, correlation: float = None, max_bins: int = 1000) -> int:
    if correlation is None or num_observations <= 1:
        return max_bins
    if correlation == 1:
        return 1
    z = (8 + 324 * num_observations + 12 * (36 * num_observations + 729 * num_observations ** 2) ** .5) ** (1 / 3)
    return int(min(round(z / 6 + 2 / (3 * z) + 1 / 3), max_bins))

def calculate_variation_of_information_extended(x: np.ndarray, y: np.ndarray, norm: bool = False) -> float:
    num_bins = calculate_number_of_bins(x.shape[0], np.corrcoef(x, y)[0, 1])
    return calculate_variation_of_information(x, y, num_bins, norm)

def calculate_mutual_information(x: np.ndarray, y: np.ndarray, norm: bool = False) -> float:
    num_bins = calculate_number_of_bins(x.shape[0], np.corrcoef(x, y)[0, 1])
    histogram_xy = np.histogram2d(x, y, num_bins)[0]
    mutual_info = mutual_info_score(None, None, contingency=histogram_xy)
    if norm:
        marginal_x = ss.entropy(np.histogram(x, num_bins)[0])
        marginal_y = ss.entropy(np.histogram(y, num_bins)[0])
        mutual_info /= min(marginal_x, marginal_y)
    return mutual_info

def exit_time(close: pd.Series, number_hours: int) -> pd.DataFrame:
    timestamps = close.index.searchsorted(close.index + pd.Timedelta(hours=number_hours))
    valid_timestamps = timestamps[timestamps < close.shape[0]]
    return pd.Series(close.index[valid_timestamps], index=close.index[:valid_timestamps.shape[0]])

def calculate_technical_features(close: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    features['frac_diff_log'] = fracdiff_log_price(close)
    features['volat'] = get_volat_w_log_returns(close, span=10)
    features['proc'] = close.pct_change(24, fill_method=None) * 100

    weights = np.arange(1, 25)
    features['wpc'] = close.rolling(24).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    bb = ta.volatility.BollingerBands(close)
    features['bb_upper'] = bb.bollinger_hband()
    features['bb_middle'] = bb.bollinger_mavg()
    features['bb_lower'] = bb.bollinger_lband()

    macd_line = close.ewm(12).mean() - close.ewm(26).mean()
    signal_line = macd_line.ewm(9).mean()
    features['macd'] = macd_line - signal_line

    features['sma_10'] = ta.trend.SMAIndicator(close, 10).sma_indicator()
    features['ema_10'] = ta.trend.EMAIndicator(close, 10).ema_indicator()

    for lag in range(1, 6):
        features[f'lag_{lag}'] = close.shift(lag)

    features['rsi_24'] = ta.momentum.RSIIndicator(close, 24).rsi()

    return features

def select_features(scaled_features: pd.DataFrame) -> pd.DataFrame:
    selected = set(scaled_features.columns)
    for i, col_i in enumerate(scaled_features.columns):
        for col_j in scaled_features.columns[i + 1:]:
            vi = calculate_variation_of_information_extended(scaled_features[col_i].values, scaled_features[col_j].values, True)
            mi = calculate_mutual_information(scaled_features[col_i].values, scaled_features[col_j].values, True)
            if vi < 0.2 or mi > 0.8:
                selected.discard(col_j)
    return scaled_features[list(selected)]

def process_currency(prices_hourly: pd.DataFrame, currency: str):
    logger.info(f"Processing {currency}")
    close = prices_hourly[currency]
    features = calculate_technical_features(close).dropna()

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), index=features.index, columns=features.columns)
    final_features = select_features(scaled_features)

    target = np.sign(close.pct_change(fill_method=None).shift(-1)).fillna(0)
    times = exit_time(close, 24).to_frame(name='end_time')
    times['start_time'] = times.index

    index = final_features.index.intersection(target.index).intersection(times.index)
    save_results(currency, final_features.loc[index], target.loc[index], close.loc[index], times.loc[index])

def save_results(currency: str, features: pd.DataFrame, target: pd.Series, close: pd.Series, times: pd.DataFrame):
    """Save processed data to the labels output directory"""
    try:
        features.to_csv(LABELS_OUTPUT_DIR / f"{currency}_features.csv")
        target.to_csv(LABELS_OUTPUT_DIR / f"{currency}_target.csv")
        close.to_csv(LABELS_OUTPUT_DIR / f"{currency}_close.csv")
        times.to_csv(LABELS_OUTPUT_DIR / f"{currency}_times.csv")
        logger.info(f"Successfully saved results for {currency}")
    except Exception as e:
        logger.error(f"Error saving results for {currency}: {str(e)}")
        raise

def load_price_data() -> pd.DataFrame:
    """Load and preprocess price data from the data directory"""
    try:
        files = list(DATA_DIR.glob('*.csv'))
        if not files:
            logger.error(f"No CSV files found in {DATA_DIR}")
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

        logger.info(f"Found {len(files)} CSV files in {DATA_DIR}")
        
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                df.set_index('Date', inplace=True)
                dfs.append(df)
                logger.debug(f"Successfully loaded {f.name}")
            except Exception as e:
                logger.error(f"Error loading {f.name}: {str(e)}")
                continue

        prices = pd.concat(dfs, axis=1)
        prices.index = pd.to_datetime(prices.index)
        return prices.resample('1h').ffill()
    
    except Exception as e:
        logger.error(f"Error in load_price_data: {str(e)}")
        raise

def main():
    logger.info("Starting labels processing for strategy 1")
    try:
        prices_hourly = load_price_data()
        logger.info(f"Processing {len(prices_hourly.columns)} currencies")
        
        for currency in prices_hourly.columns:
            logger.info(f"Processing {currency}")
            process_currency(prices_hourly, currency)
            
        logger.info("Labels processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()