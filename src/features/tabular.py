"""
Extract ~25 financial ratio features from price data.

Input: data/raw/prices/*.csv (one per ticker)
Output: DataFrame with columns [ticker, quarter, feature_1, ..., feature_25]
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_prices_data(data_dir: str = "data/raw/prices") -> dict:
    """
    Load all price CSVs into a dict keyed by ticker.
    
    Returns:
        dict: {ticker: pd.DataFrame with columns [date, open, high, low, close, volume, reported_eps, eps_estimate]}
    """
    prices_dict = {}
    data_path = Path(data_dir)
    
    for csv_file in data_path.glob("*.csv"):
        ticker = csv_file.stem  # filename without extension (e.g., "AAPL")
        df = pd.read_csv(csv_file, parse_dates=['date'])
        df.set_index('date', inplace=True)
        prices_dict[ticker] = df
    
    return prices_dict


def compute_quarterly_features(prices_df: pd.DataFrame, ticker: str) -> list:
    """
    Given a price DataFrame for one ticker, compute quarterly financial features.
    
    Args:
        prices_df: DataFrame indexed by date, has columns: close, open, high, low, volume, reported_eps, eps_estimate
        ticker: ticker symbol
    
    Returns:
        list of dicts with quarterly features
    """
    results = []
    
    # Create a copy and add quarter/year columns
    df = prices_df.copy()
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    
    # Compute log returns for volatility
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Compute surprise: (actual - consensus) / abs(consensus)
    # Handle cases where eps_estimate is null or 0
    df['surprise'] = (df['reported_eps'] - df['eps_estimate']) / np.abs(df['eps_estimate']).replace(0, np.nan)
    
    # Group by quarter
    for quarter, group in df.groupby('year_quarter'):
        if len(group) < 2:  # Skip quarters with < 2 data points
            continue
        
        # Initialize feature dict
        feature_dict = {
            'ticker': ticker,
            'quarter': quarter,
        }
        
        # 1. Prior quarter surprise (last day's surprise)
        feature_dict['surprise_latest'] = group['surprise'].iloc[-1] if not group['surprise'].isna().all() else 0
        
        # 2. Price return over quarter
        price_return = (group['close'].iloc[-1] - group['close'].iloc[0]) / group['close'].iloc[0]
        feature_dict['price_return'] = price_return
        
        # 3. Volatility (std of log returns)
        volatility = group['log_return'].std()
        feature_dict['volatility'] = volatility if not np.isnan(volatility) else 0
        
        # 4. Average volume
        avg_volume = group['volume'].mean()
        feature_dict['avg_volume'] = avg_volume
        
        # 5. Earnings yield (EPS / price)
        eps_latest = group['reported_eps'].iloc[-1]
        price_latest = group['close'].iloc[-1]
        earnings_yield = eps_latest / price_latest if price_latest != 0 and pd.notna(eps_latest) else 0
        feature_dict['earnings_yield'] = earnings_yield
        
        # 6. EPS growth rate
        eps_first = group['reported_eps'].iloc[0]
        eps_growth = (eps_latest - eps_first) / np.abs(eps_first) if eps_first != 0 and pd.notna(eps_latest) and pd.notna(eps_first) else 0
        feature_dict['eps_growth'] = eps_growth
        
        # 7. Consensus accuracy
        consensus_eps = group['eps_estimate'].iloc[-1]
        actual_eps = group['reported_eps'].iloc[-1]
        consensus_error = (actual_eps - consensus_eps) / np.abs(consensus_eps) if consensus_eps != 0 and pd.notna(actual_eps) and pd.notna(consensus_eps) else 0
        feature_dict['consensus_error'] = consensus_error
        
        # 8. Price momentum
        momentum = (group['close'].iloc[-1] - group['close'].iloc[0]) / group['close'].iloc[0]
        feature_dict['momentum'] = momentum
        
        # 9. High-low range (intra-quarter volatility)
        hl_range = (group['high'].max() - group['low'].min()) / group['close'].mean()
        feature_dict['hl_range'] = hl_range
        
        # 10. Volume trend
        volume_trend = group['volume'].iloc[-1] / group['volume'].iloc[0] if group['volume'].iloc[0] != 0 else 1
        feature_dict['volume_trend'] = volume_trend
        
        # 11. Close to high ratio
        close_to_high = group['close'].iloc[-1] / group['high'].max() if group['high'].max() != 0 else 1
        feature_dict['close_to_high'] = close_to_high
        
        # 12. Close to low ratio
        close_to_low = group['close'].iloc[-1] / group['low'].min() if group['low'].min() != 0 else 1
        feature_dict['close_to_low'] = close_to_low
        
        # 13. Intra-quarter high return
        max_return = (group['high'].max() - group['close'].iloc[0]) / group['close'].iloc[0]
        feature_dict['max_return'] = max_return
        
        # 14. Intra-quarter low return
        min_return = (group['low'].min() - group['close'].iloc[0]) / group['close'].iloc[0]
        feature_dict['min_return'] = min_return
        
        # 15. EPS consistency
        eps_std = group['reported_eps'].std()
        feature_dict['eps_std'] = eps_std if not np.isnan(eps_std) else 0
        
        # 16. Consensus convergence
        consensus_std = group['eps_estimate'].std()
        feature_dict['consensus_std'] = consensus_std if not np.isnan(consensus_std) else 0
        
        # 17. Price-to-consensus
        price_to_consensus = price_latest / consensus_eps if consensus_eps != 0 else 0
        feature_dict['price_to_consensus'] = price_to_consensus
        
        # 18. Trading days
        feature_dict['trading_days'] = len(group)
        
        # 19. Average daily return
        avg_daily_return = group['log_return'].mean()
        feature_dict['avg_daily_return'] = avg_daily_return if not np.isnan(avg_daily_return) else 0
        
        # 20. Sharpe-like ratio
        sharpe_proxy = avg_daily_return / volatility if volatility > 0 else 0
        feature_dict['sharpe_proxy'] = sharpe_proxy if not np.isnan(sharpe_proxy) else 0
        
        # 21. EPS to close ratio
        eps_to_close = eps_latest / price_latest if price_latest != 0 else 0
        feature_dict['eps_to_close'] = eps_to_close
        
        # 22. Consensus to actual EPS ratio
        consensus_to_actual = consensus_eps / actual_eps if actual_eps != 0 else 0
        feature_dict['consensus_to_actual'] = consensus_to_actual
        
        # 23. Quarter-end gap
        last_day = group.iloc[-1]
        gap = (last_day['close'] - last_day['open']) / last_day['open'] if last_day['open'] != 0 else 0
        feature_dict['quarter_end_gap'] = gap
        
        # 24. Volume weighted price
        vwap = (group['close'] * group['volume']).sum() / group['volume'].sum() if group['volume'].sum() != 0 else 0
        feature_dict['vwap'] = vwap
        
        # 25. Distance from VWAP
        distance_from_vwap = (price_latest - vwap) / vwap if vwap != 0 else 0
        feature_dict['distance_from_vwap'] = distance_from_vwap
        
        results.append(feature_dict)
    
    return results


def generate_tabular_features(data_dir: str = "data/raw/prices") -> pd.DataFrame:
    """
    Load all price data, compute quarterly features for each ticker, return merged DataFrame.
    
    Returns:
        pd.DataFrame: columns [ticker, quarter, feature_1, ..., feature_25]
    """
    prices_dict = load_prices_data(data_dir)
    all_features = []
    
    for ticker, prices_df in prices_dict.items():
        features = compute_quarterly_features(prices_df, ticker)
        all_features.extend(features)
    
    result_df = pd.DataFrame(all_features)
    return result_df


if __name__ == "__main__":
    tabular_df = generate_tabular_features()
    print(f"Generated tabular features: {tabular_df.shape}")
    print(tabular_df.head())
