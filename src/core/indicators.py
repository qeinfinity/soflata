# VolDelta-MR: src/core/indicators.py
# Contains functions for calculating all trading indicators.
# (e.g., Volume Delta Z-Score, VWAP Bands, ATR)

import pandas as pd
import numpy as np

# Placeholder for calculate_volume_delta_zscore (detailed in PRD)
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def calculate_volume_delta_zscore(df_1min_data_input, ticks_win=60, sig_len=7, cmp_len=14, z_len=100):
    # Work on a copy to avoid SettingWithCopyWarning on the input DataFrame
    df_1min_data = df_1min_data_input.copy()

    if 'hlc3' not in df_1min_data.columns:
        df_1min_data['hlc3'] = (df_1min_data['high'] + df_1min_data['low'] + df_1min_data['close']) / 3

    df_1min_data['momentary_delta'] = (df_1min_data['hlc3'] - df_1min_data['hlc3'].shift(1)) * df_1min_data['volume']
    
    # FIX for Pandas Warning: Use .loc for assignment
    if not df_1min_data.empty: # Ensure DataFrame is not empty before accessing iloc[0]
        df_1min_data.loc[df_1min_data.index[0], 'momentary_delta'] = 0 # Assign to the first row directly

    # ... rest of the function remains the same ...
    df_1min_data['roll'] = df_1min_data['momentary_delta'].rolling(window=ticks_win, min_periods=1).sum()

    if not isinstance(df_1min_data.index, pd.DatetimeIndex):
        if 'timestamp' in df_1min_data.columns:
             df_1min_data['timestamp'] = pd.to_datetime(df_1min_data['timestamp'])
             df_1min_data.set_index('timestamp', inplace=True)
        else: # Attempt to convert index if it's not datetime but also not 'timestamp' column
             try:
                df_1min_data.index = pd.to_datetime(df_1min_data.index)
             except Exception as e:
                print(f"[ERROR] Could not convert index to DatetimeIndex in Z-score calc: {e}")
                return pd.DataFrame()


    df_30min = pd.DataFrame()
    df_30min['sampled_roll'] = df_1min_data['roll'].resample('30min').last()
    df_30min['close'] = df_1min_data['close'].resample('30min').last() # Also bring over 30min close
    df_30min.dropna(subset=['sampled_roll'], inplace=True)

    if df_30min.empty:
        print("[WARN] df_30min is empty after resampling and dropna in Z-score calculation.")
        return pd.DataFrame({'volume_delta_zscore': []}) # Return empty DF with expected column

    df_30min['fast_ema'] = df_30min['sampled_roll'].ewm(span=sig_len, adjust=False, min_periods=sig_len).mean()
    df_30min['slow_ema'] = df_30min['sampled_roll'].ewm(span=cmp_len, adjust=False, min_periods=cmp_len).mean()
    df_30min['hist'] = df_30min['fast_ema'] - df_30min['slow_ema']

    df_30min['hist_mu'] = df_30min['hist'].rolling(window=z_len, min_periods=max(1, z_len // 2)).mean()
    df_30min['hist_sigma'] = df_30min['hist'].rolling(window=z_len, min_periods=max(1, z_len // 2)).std()
    
    df_30min['volume_delta_zscore'] = np.where(
        df_30min['hist_sigma'].isnull() | (df_30min['hist_sigma'] == 0), 
        0, 
        (df_30min['hist'] - df_30min['hist_mu']) / df_30min['hist_sigma']
    )
    # Fill any remaining NaNs in z-score (e.g. if hist_mu was NaN) with 0
    df_30min['volume_delta_zscore'].fillna(0, inplace=True)

    return df_30min # Return full df for now



# Placeholder for calculate_session_vwap_bands (detailed in PRD)
def calculate_session_vwap_bands(df_30min_data):
    # Ensure timestamp is index and datetime
    if not isinstance(df_30min_data.index, pd.DatetimeIndex):
        # Assuming 'timestamp' column exists if not index
        df_30min_data['timestamp'] = pd.to_datetime(df_30min_data['timestamp'])
        df_30min_data.set_index('timestamp', inplace=True)
    
    df = df_30min_data.copy() # Work on a copy
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_volume'] = df['typical_price'] * df['volume']

    # Group by date to handle session resets (UTC 00:00)
    # Note: .date might not work directly in groupby for cumsum if index is DatetimeIndex.
    # A common way is to create a 'date' column first.
    df['date_group'] = df.index.normalize() # Normalize to midnight UTC

    df['cumulative_price_volume'] = df.groupby('date_group')['price_volume'].cumsum()
    df['cumulative_volume'] = df.groupby('date_group')['volume'].cumsum()
    
    df['session_vwap'] = df['cumulative_price_volume'] / df['cumulative_volume']
    
    # Standard Deviation Calculation for VWAP Bands
    # Variance = Sum((Price - VWAP)^2 * Volume) / Sum(Volume) for the session
    df['price_minus_vwap_sq'] = (df['typical_price'] - df['session_vwap'])**2
    df['price_minus_vwap_sq_vol'] = df['price_minus_vwap_sq'] * df['volume']
    
    df['cumulative_price_minus_vwap_sq_vol'] = df.groupby('date_group')['price_minus_vwap_sq_vol'].cumsum()
    
    # Ensure cumulative_volume is not zero to avoid division by zero
    df['variance'] = np.where(df['cumulative_volume'] == 0, 0, \
                            df['cumulative_price_minus_vwap_sq_vol'] / df['cumulative_volume'])
    df['session_stdev'] = np.sqrt(df['variance'])

    # Calculate Bands (Example for ±1, ±2, ±3 sigma)
    df['vwap_upper1'] = df['session_vwap'] + (1 * df['session_stdev'])
    df['vwap_lower1'] = df['session_vwap'] - (1 * df['session_stdev'])
    df['vwap_upper2'] = df['session_vwap'] + (2 * df['session_stdev'])
    df['vwap_lower2'] = df['session_vwap'] - (2 * df['session_stdev'])
    df['vwap_upper3'] = df['session_vwap'] + (3 * df['session_stdev'])
    df['vwap_lower3'] = df['session_vwap'] - (3 * df['session_stdev'])
    
    # Select relevant columns to return or return full df
    # relevant_cols = ['open', 'high', 'low', 'close', 'volume', 'session_vwap', 
    #                  'session_stdev', 'vwap_upper1', 'vwap_lower1', 
    #                  'vwap_upper2', 'vwap_lower2', 'vwap_upper3', 'vwap_lower3']
    # return df[relevant_cols]
    return df # Return full df for flexibility

# In src/core/indicators.py, add:
def calculate_atr(df, period=20):
    # df should have 'high', 'low', 'close' columns
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_close_prev, 'lpc': low_close_prev}).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean() # Wilder's ATR often uses EMA like this
    # or for simple ATR: atr = tr.rolling(window=period).mean()
    return atr

# Then, when creating df_master_30min:
# df_master_30min['atr_20_30m'] = calculate_atr(df_master_30min, period=20)

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

