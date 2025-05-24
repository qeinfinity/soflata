# VolDelta-MR: src/core/indicators.py
# Contains functions for calculating all trading indicators.
# (e.g., Volume Delta Z-Score, VWAP Bands, ATR)

import pandas as pd
import numpy as np

# Placeholder for calculate_volume_delta_zscore (detailed in PRD)
import pandas as pd
import numpy as np

def calculate_volume_delta_zscore(df_1min_data, ticks_win=60, sig_len=7, cmp_len=14, z_len=100):
    # Ensure 'hlc3' exists or calculate it
    if 'hlc3' not in df_1min_data.columns:
        df_1min_data['hlc3'] = (df_1min_data['high'] + df_1min_data['low'] + df_1min_data['close']) / 3

    # Step 1: Raw Momentum Slices
    df_1min_data['momentary_delta'] = (df_1min_data['hlc3'] - df_1min_data['hlc3'].shift(1)) * df_1min_data['volume']
    df_1min_data['momentary_delta'].iloc[0] = 0 # Handle first NaN

    # Step 2: Rolling Momentum Sum
    df_1min_data['roll'] = df_1min_data['momentary_delta'].rolling(window=ticks_win, min_periods=1).sum() # min_periods=1 to get values earlier

    # Step 3: Resample to 30-Min Bars & Sample "roll"
    # Ensure timestamp is index and datetime object
    if not isinstance(df_1min_data.index, pd.DatetimeIndex):
        df_1min_data['timestamp'] = pd.to_datetime(df_1min_data['timestamp'])
        df_1min_data.set_index('timestamp', inplace=True)

    df_30min = pd.DataFrame()
    # Sample 'roll' using the last value in each 30min interval
    df_30min['sampled_roll'] = df_1min_data['roll'].resample('30min').last()
    # Also, bring over the 30-min close for context if needed later
    df_30min['close'] = df_1min_data['close'].resample('30min').last()
    df_30min.dropna(subset=['sampled_roll'], inplace=True) # Remove rows where sampled_roll is NaN (beginning of dataset)


    # Step 4: Calculate Momentum Histogram
    df_30min['fast_ema'] = df_30min['sampled_roll'].ewm(span=sig_len, adjust=False, min_periods=sig_len).mean()
    df_30min['slow_ema'] = df_30min['sampled_roll'].ewm(span=cmp_len, adjust=False, min_periods=cmp_len).mean()
    df_30min['hist'] = df_30min['fast_ema'] - df_30min['slow_ema']

    # Step 5: Calculate Z-Score
    df_30min['hist_mu'] = df_30min['hist'].rolling(window=z_len, min_periods=z_len // 2).mean() # Allow fewer periods at start
    df_30min['hist_sigma'] = df_30min['hist'].rolling(window=z_len, min_periods=z_len // 2).std()
    
    # Calculate Z-Score, ensuring sigma is not zero
    df_30min['volume_delta_zscore'] = np.where(df_30min['hist_sigma'] == 0, 0, \
                                        (df_30min['hist'] - df_30min['hist_mu']) / df_30min['hist_sigma'])
    
    # Clean up intermediate columns if you only want the Z-score and close
    # For debugging, you might want to keep them.
    # return df_30min[['close', 'volume_delta_zscore']]
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

