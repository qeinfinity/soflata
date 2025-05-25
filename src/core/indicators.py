# File: src/core/indicators.py

import pandas as pd
import numpy as np
import math # For math.isclose if needed later, not directly in this func

def calculate_volume_delta_zscore(df_1min_data_input, ticks_win=60, sig_len=7, cmp_len=14, z_len=100):
    """
    Calculates the Volume Delta Z-Score.
    Takes 1-minute DataFrame, returns 30-minute DataFrame with Z-Score and intermediate calcs.
    """
    if df_1min_data_input.empty:
        print("[WARN] Input 1-min data for Z-score is empty.")
        return pd.DataFrame(columns=['volume_delta_zscore', 'sampled_roll', 'close']) # Return with expected columns

    df_1min_data = df_1min_data_input.copy() # Work on a copy

    if 'hlc3' not in df_1min_data.columns:
        df_1min_data['hlc3'] = (df_1min_data['high'] + df_1min_data['low'] + df_1min_data['close']) / 3

    df_1min_data['momentary_delta'] = (df_1min_data['hlc3'] - df_1min_data['hlc3'].shift(1)) * df_1min_data['volume']
    
    if not df_1min_data.empty:
        df_1min_data.loc[df_1min_data.index[0], 'momentary_delta'] = 0.0 # Use .loc for safe assignment

    df_1min_data['roll'] = df_1min_data['momentary_delta'].rolling(window=ticks_win, min_periods=1).sum()

    # Ensure index is DatetimeIndex before resampling
    if not isinstance(df_1min_data.index, pd.DatetimeIndex):
        if 'timestamp' in df_1min_data.columns: # Check if timestamp column exists
            try:
                df_1min_data['timestamp'] = pd.to_datetime(df_1min_data['timestamp'])
                df_1min_data.set_index('timestamp', inplace=True)
            except Exception as e:
                print(f"[ERROR] Failed to set datetime index from 'timestamp' column in Z-score: {e}")
                return pd.DataFrame(columns=['volume_delta_zscore', 'sampled_roll', 'close'])
        else: # Try to convert existing index
            try:
                df_1min_data.index = pd.to_datetime(df_1min_data.index)
            except Exception as e:
                print(f"[ERROR] Failed to convert existing index to DatetimeIndex in Z-score: {e}")
                return pd.DataFrame(columns=['volume_delta_zscore', 'sampled_roll', 'close'])
    
    # Resample to 30-min bars
    df_30min = pd.DataFrame()
    # Ensure there's data to resample to avoid errors with empty df_1min_data['roll']
    if df_1min_data.empty or 'roll' not in df_1min_data.columns:
         print("[WARN] Not enough data or 'roll' column missing for 30min resampling in Z-score.")
         return pd.DataFrame(columns=['volume_delta_zscore', 'sampled_roll', 'close'])

    df_30min['sampled_roll'] = df_1min_data['roll'].resample('30min').last()
    df_30min['close'] = df_1min_data['close'].resample('30min').last() # Also bring over 30min close
    
    # Option for handling potential NaNs in sampled_roll if 1-min data has gaps within 30-min windows
    # df_30min['sampled_roll'].ffill(inplace=True) # Forward fill, use with caution, understand implications.

    df_30min.dropna(subset=['sampled_roll', 'close'], inplace=True) # Ensure both roll and close are non-NaN for a valid 30-min bar

    if df_30min.empty:
        print("[WARN] df_30min is empty after resampling and dropna in Z-score calculation.")
        return pd.DataFrame(columns=['volume_delta_zscore', 'sampled_roll', 'close'])

    # Calculate Momentum Histogram
    # Use min_periods to get EMAs sooner, important if data length is short
    df_30min['fast_ema'] = df_30min['sampled_roll'].ewm(span=sig_len, adjust=False, min_periods=max(1,sig_len//2)).mean()
    df_30min['slow_ema'] = df_30min['sampled_roll'].ewm(span=cmp_len, adjust=False, min_periods=max(1,cmp_len//2)).mean()
    df_30min['hist'] = df_30min['fast_ema'] - df_30min['slow_ema']

    # Calculate Z-Score
    df_30min['hist_mu'] = df_30min['hist'].rolling(window=z_len, min_periods=max(1, z_len // 2)).mean()
    df_30min['hist_sigma'] = df_30min['hist'].rolling(window=z_len, min_periods=max(1, z_len // 2)).std()
    
    df_30min['volume_delta_zscore'] = np.where(
        df_30min['hist_sigma'].isnull() | (df_30min['hist_sigma'] == 0), 
        0.0, # Default Z-score to 0 if sigma is NaN or zero
        (df_30min['hist'] - df_30min['hist_mu']) / df_30min['hist_sigma']
    )
    df_30min['volume_delta_zscore'].fillna(0.0, inplace=True) # Final catch-all for any other NaNs

    # Return relevant columns, including 'close' for merging and 'sampled_roll' if needed for other calcs/debug
    return df_30min[['close', 'sampled_roll', 'volume_delta_zscore', 'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma']]


def calculate_session_vwap_bands(df_30min_data_input):
    if df_30min_data_input.empty:
        print("[WARN] Input 30-min data for VWAP is empty.")
        return df_30min_data_input # Return empty to avoid errors

    df = df_30min_data_input.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            except Exception as e:
                print(f"[ERROR] Failed to set datetime index from 'timestamp' column in VWAP: {e}")
                return df_30min_data_input
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"[ERROR] Failed to convert existing index to DatetimeIndex in VWAP: {e}")
                return df_30min_data_input
    
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['price_volume'] = df['typical_price'] * df['volume']

    df['date_group'] = df.index.normalize() # Normalize to midnight UTC for session grouping

    df['cumulative_price_volume'] = df.groupby('date_group')['price_volume'].cumsum()
    df['cumulative_volume'] = df.groupby('date_group')['volume'].cumsum()
    
    # Handle division by zero for session_vwap
    df['session_vwap'] = np.where(
        df['cumulative_volume'] == 0, 
        np.nan, # Or could use previous VWAP with ffill later, but NaN is safer first
        df['cumulative_price_volume'] / df['cumulative_volume']
    )
    # Forward fill VWAP for initial bars where cum_vol might be 0 or data sparse
    df['session_vwap'].ffill(inplace=True)
        
    df['price_minus_vwap_sq'] = (df['typical_price'] - df['session_vwap'])**2.0 # Ensure float
    df['price_minus_vwap_sq_vol'] = df['price_minus_vwap_sq'] * df['volume']
    
    df['cumulative_price_minus_vwap_sq_vol'] = df.groupby('date_group')['price_minus_vwap_sq_vol'].cumsum()
    
    df['variance'] = np.where(
        df['cumulative_volume'] == 0, 
        0.0, # Variance is 0 if no volume 
        df['cumulative_price_minus_vwap_sq_vol'] / df['cumulative_volume']
    )
    df['session_stdev'] = np.sqrt(df['variance'])
    # Forward fill stdev if needed (e.g. first bar of session might have NaN stdev if VWAP was NaN)
    df['session_stdev'].ffill(inplace=True)
    df['session_stdev'].fillna(0.0, inplace=True) # Fill any remaining NaNs (e.g. if VWAP was persistently NaN)

    # Calculate Bands
    for i in [1, 2, 3]:
        df[f'vwap_upper{i}'] = df['session_vwap'] + (i * df['session_stdev'])
        df[f'vwap_lower{i}'] = df['session_vwap'] - (i * df['session_stdev'])
    
    return df


def calculate_atr(df_input, period=20):
    if df_input.empty or not all(col in df_input.columns for col in ['high', 'low', 'close']):
        print("[WARN] Input data for ATR is empty or missing required columns.")
        return pd.Series(index=df_input.index, dtype=float) # Return empty Series or Series of NaNs

    df = df_input.copy()
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    tr_df = pd.DataFrame({'hl': high_low, 'hpc': high_close_prev, 'lpc': low_close_prev})
    tr = tr_df.max(axis=1)
    # Use Wilder's smoothing method (alpha = 1/period)
    atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=max(1,period//2)).mean() 
    return atr

if __name__ == '__main__':
    print("Testing indicator functions (placeholders)...")
    # Add simple test calls here if desired for standalone testing
    dummy_data = pd.DataFrame(np.random.rand(100,1), columns=['close'], index=pd.date_range('2023-01-01', periods=100, freq='1min'))
    dummy_data['high'] = dummy_data['close'] * 1.02
    dummy_data['low'] = dummy_data['close'] * 0.98
    dummy_data['open'] = dummy_data['close']
    dummy_data['volume'] = np.random.randint(100,1000, size=100)

    z_score_df = calculate_volume_delta_zscore(dummy_data.copy())
    print("Z-Score DF head:\n", z_score_df.head())
    
    # For VWAP and ATR, we'd typically use 30-min data, so resample first for dummy
    df_30m_dummy = dummy_data.resample('30min').agg({
        'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
    }).dropna()

    if not df_30m_dummy.empty:
        vwap_df = calculate_session_vwap_bands(df_30m_dummy.copy())
        print("VWAP DF head:\n", vwap_df.head())
        atr_series = calculate_atr(df_30m_dummy.copy())
        print("ATR Series head:\n", atr_series.head())
    else:
        print("Not enough data for 30min dummy resampling.")

