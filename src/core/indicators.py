# File: src/core/indicators.py

import pandas as pd
import numpy as np
import math # For math.isclose if needed later

def calculate_volume_delta_zscore(df_1min_data_input, ticks_win=60, sig_len=7, cmp_len=14, z_len=100):
    """
    Calculates the Volume Delta Z-Score.
    Takes 1-minute DataFrame, returns 30-minute DataFrame with Z-Score and intermediate calcs.
    """
    if df_1min_data_input.empty:
        print("[WARN] Input 1-min data for Z-score is empty.")
        # Return a DataFrame with expected columns for graceful failure downstream
        return pd.DataFrame(columns=['close', 'sampled_roll', 'volume_delta_zscore', 
                                     'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma'])

    df_1min_data = df_1min_data_input.copy() # Work on a copy

    # Ensure 'hlc3' exists or calculate it
    if 'hlc3' not in df_1min_data.columns:
        if not all(col in df_1min_data.columns for col in ['high', 'low', 'close']):
            print("[ERROR] Missing 'high', 'low', or 'close' columns for hlc3 calculation in Z-score.")
            return pd.DataFrame(columns=['close', 'sampled_roll', 'volume_delta_zscore', 
                                         'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma'])
        df_1min_data['hlc3'] = (df_1min_data['high'] + df_1min_data['low'] + df_1min_data['close']) / 3.0

    # Step 1: Raw Momentum Slices
    df_1min_data['momentary_delta'] = (df_1min_data['hlc3'] - df_1min_data['hlc3'].shift(1)) * df_1min_data['volume']
    
    # FIX for Pandas Warning: Use .loc for assignment to the first row
    if not df_1min_data.empty: 
        df_1min_data.loc[df_1min_data.index[0], 'momentary_delta'] = 0.0 # Assign 0.0 to the first row directly

    # Step 2: Rolling Momentum Sum
    # min_periods=1 to get values earlier, helps with shorter datasets at the start
    df_1min_data['roll'] = df_1min_data['momentary_delta'].rolling(window=ticks_win, min_periods=1).sum()

    # Step 3: Resample to 30-Min Bars & Sample "roll"
    # Ensure timestamp is index and datetime object
    if not isinstance(df_1min_data.index, pd.DatetimeIndex):
        if 'timestamp' in df_1min_data.columns:
            try:
                df_1min_data['timestamp'] = pd.to_datetime(df_1min_data['timestamp'])
                df_1min_data.set_index('timestamp', inplace=True)
            except Exception as e:
                print(f"[ERROR] Failed to set datetime index from 'timestamp' column in Z-score: {e}")
                return pd.DataFrame(columns=['close', 'sampled_roll', 'volume_delta_zscore', 
                                             'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma'])
        else: 
            try:
                df_1min_data.index = pd.to_datetime(df_1min_data.index)
            except Exception as e:
                print(f"[ERROR] Failed to convert existing index to DatetimeIndex in Z-score: {e}")
                return pd.DataFrame(columns=['close', 'sampled_roll', 'volume_delta_zscore', 
                                             'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma'])
    
    df_30min = pd.DataFrame()
    if df_1min_data.empty or 'roll' not in df_1min_data.columns or 'close' not in df_1min_data.columns:
         print("[WARN] Not enough data or 'roll'/'close' column missing for 30min resampling in Z-score.")
         return pd.DataFrame(columns=['close', 'sampled_roll', 'volume_delta_zscore', 
                                      'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma'])

    df_30min['sampled_roll'] = df_1min_data['roll'].resample('30min').last()
    df_30min['close'] = df_1min_data['close'].resample('30min').last() 
    
    # Option: Forward fill sampled_roll if gaps in 1-min data cause NaNs after .last()
    # This assumes that if a 30-min window has no 1-min data at its exact end,
    # the last known roll value from a previous 1-min interval is a reasonable proxy.
    # Use with caution and understand the implication for your data.
    # df_30min['sampled_roll'] = df_30min['sampled_roll'].ffill()

    df_30min.dropna(subset=['sampled_roll', 'close'], inplace=True) # Ensure valid bar start

    if df_30min.empty:
        print("[WARN] df_30min is empty after resampling and initial dropna in Z-score calculation.")
        return pd.DataFrame(columns=['close', 'sampled_roll', 'volume_delta_zscore', 
                                     'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma'])

    # Step 4: Calculate Momentum Histogram
    # Using min_periods to allow EMAs to calculate with fewer than full span initially
    df_30min['fast_ema'] = df_30min['sampled_roll'].ewm(span=sig_len, adjust=False, min_periods=max(1, sig_len // 2)).mean()
    df_30min['slow_ema'] = df_30min['sampled_roll'].ewm(span=cmp_len, adjust=False, min_periods=max(1, cmp_len // 2)).mean()
    df_30min['hist'] = df_30min['fast_ema'] - df_30min['slow_ema']

    # Step 5: Calculate Z-Score
    df_30min['hist_mu'] = df_30min['hist'].rolling(window=z_len, min_periods=max(1, z_len // 2)).mean()
    df_30min['hist_sigma'] = df_30min['hist'].rolling(window=z_len, min_periods=max(1, z_len // 2)).std()
    
    # Calculate Z-Score, ensuring sigma is not zero or NaN
    df_30min['volume_delta_zscore'] = np.where(
        df_30min['hist_sigma'].isnull() | (df_30min['hist_sigma'] == 0), 
        0.0, # Default Z-score to 0 if sigma is NaN or zero
        (df_30min['hist'] - df_30min['hist_mu']) / df_30min['hist_sigma']
    )
    # FIX for inplace FutureWarning: Assign back to the column
    df_30min['volume_delta_zscore'] = df_30min['volume_delta_zscore'].fillna(0.0)

    # Return all calculated columns for potential debugging or advanced use, including 'close' for merging.
    return df_30min[['close', 'sampled_roll', 'volume_delta_zscore', 
                     'fast_ema', 'slow_ema', 'hist', 'hist_mu', 'hist_sigma']]


def calculate_session_vwap_bands(df_30min_data_input):
    """
    Calculates Session VWAP and Standard Deviation Bands on 30-minute data.
    """
    if df_30min_data_input.empty:
        print("[WARN] Input 30-min data for VWAP is empty.")
        # Return a DataFrame with expected columns for graceful failure
        cols = ['session_vwap', 'session_stdev'] + [f'vwap_upper{i}' for i in [1,2,3]] + [f'vwap_lower{i}' for i in [1,2,3]]
        # Preserve original columns if possible
        final_cols = list(df_30min_data_input.columns) + [c for c in cols if c not in df_30min_data_input.columns]
        return pd.DataFrame(columns=final_cols, index=df_30min_data_input.index)


    df = df_30min_data_input.copy() # Work on a copy

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            except Exception as e:
                print(f"[ERROR] Failed to set datetime index from 'timestamp' column in VWAP: {e}")
                return df_30min_data_input # Return original on error to prevent downstream issues
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"[ERROR] Failed to convert existing index to DatetimeIndex in VWAP: {e}")
                return df_30min_data_input
    
    if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        print("[ERROR] Missing OHLCV columns for VWAP calculation.")
        return df_30min_data_input

    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['price_volume'] = df['typical_price'] * df['volume']

    # Group by date to handle session resets (UTC 00:00)
    df['date_group'] = df.index.normalize() # Ensures grouping by day at UTC midnight

    df['cumulative_price_volume'] = df.groupby('date_group')['price_volume'].cumsum()
    df['cumulative_volume'] = df.groupby('date_group')['volume'].cumsum()
    
    # Handle division by zero for session_vwap
    df['session_vwap'] = np.where(
        df['cumulative_volume'] == 0, 
        np.nan, 
        df['cumulative_price_volume'] / df['cumulative_volume']
    )
    # FIX for inplace FutureWarning: Assign back
    df['session_vwap'] = df['session_vwap'].ffill() # Forward fill for session start if first bar has no volume
        
    # Standard Deviation Calculation
    df['price_minus_vwap_sq'] = (df['typical_price'] - df['session_vwap'])**2.0 # Ensure float
    df['price_minus_vwap_sq_vol'] = df['price_minus_vwap_sq'] * df['volume']
    
    df['cumulative_price_minus_vwap_sq_vol'] = df.groupby('date_group')['price_minus_vwap_sq_vol'].cumsum()
    
    df['variance'] = np.where(
        df['cumulative_volume'] == 0, 
        0.0, # Variance is 0 if no volume for the session yet
        df['cumulative_price_minus_vwap_sq_vol'] / df['cumulative_volume']
    )
    df['session_stdev'] = np.sqrt(df['variance'])

    # FIX for inplace FutureWarnings: Assign back
    df['session_stdev'] = df['session_stdev'].ffill() # Forward fill for session start
    df['session_stdev'] = df['session_stdev'].fillna(0.0) # Fill any remaining NaNs (e.g., if VWAP itself was NaN)

    # Calculate Bands
    for i in [1, 2, 3]:
        df[f'vwap_upper{i}'] = df['session_vwap'] + (i * df['session_stdev'])
        df[f'vwap_lower{i}'] = df['session_vwap'] - (i * df['session_stdev'])
    
    # Drop the temporary grouping column
    df.drop(columns=['date_group', 'typical_price', 'price_volume', 
                     'cumulative_price_volume', 'cumulative_volume',
                     'price_minus_vwap_sq', 'price_minus_vwap_sq_vol',
                     'cumulative_price_minus_vwap_sq_vol', 'variance'], inplace=True, errors='ignore')
    
    return df


def calculate_atr(df_input, period=20):
    """
    Calculates Average True Range (ATR) using Wilder's smoothing (EMA-like).
    """
    if df_input.empty or not all(col in df_input.columns for col in ['high', 'low', 'close']):
        print("[WARN] Input data for ATR is empty or missing required HLC columns.")
        # Return a Series of NaNs with the same index to prevent merge issues
        return pd.Series(index=df_input.index, dtype=float, name=f'atr_{period}')

    df = df_input.copy() # Work on a copy

    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = np.abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = np.abs(df['low'] - df['close'].shift(1))
    
    # True Range
    # The .fillna(0) for h-pc and l-pc handles the first row's NaN from shift(1)
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1, skipna=False)
    df.loc[df.index[0], 'tr'] = df.loc[df.index[0], 'h-l'] # First TR is just High - Low

    # Wilder's ATR (uses a specific type of EMA/SMA for initialization)
    # For simplicity and common practice, an EWM directly on TR is often used.
    # Using min_periods to get values sooner. adjust=False for Wilder's method.
    # PRD review mentioned: "Wilder’s ATR uses α = 1/period applied on True Range SMA seed. You jump straight to EMA."
    # This implementation uses EWM directly which is a common approximation.
    # For a stricter Wilder's ATR:
    # sma_tr = df['tr'].rolling(window=period, min_periods=period).mean()
    # atr = pd.Series(index=df.index, dtype=float)
    # atr.iloc[period-1] = sma_tr.iloc[period-1]
    # for i in range(period, len(df)):
    #    atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + df['tr'].iloc[i]) / period
    # This is more complex. The EWM is a widely accepted alternative for backtesting.
    
    atr = df['tr'].ewm(alpha=1.0/period, adjust=False, min_periods=max(1, period // 2)).mean() 
    
    return atr.rename(f'atr_{period}') # Return Series with name

if __name__ == '__main__':
    print("--- Testing Indicator Functions (src/core/indicators.py) ---")
    
    # Create more comprehensive dummy data for testing edge cases
    index_1m = pd.date_range('2023-01-01 00:00:00', periods=120 * 3, freq='1min', tz='UTC') # 3 full 30-min bars worth for Z-score warmup
    data_1m = {
        'open': np.random.uniform(16000, 17000, len(index_1m)),
        'high': np.random.uniform(16000, 17000, len(index_1m)),
        'low': np.random.uniform(16000, 17000, len(index_1m)),
        'close': np.random.uniform(16000, 17000, len(index_1m)),
        'volume': np.random.randint(10, 1000, len(index_1m))
    }
    dummy_1min_df = pd.DataFrame(data_1m, index=index_1m)
    dummy_1min_df['high'] = dummy_1min_df[['open','high','low','close']].max(axis=1) + np.random.rand(len(index_1m)) * 50
    dummy_1min_df['low'] = dummy_1min_df[['open','high','low','close']].min(axis=1) - np.random.rand(len(index_1m)) * 50
    # Ensure low <= open,close <= high
    dummy_1min_df['low'] = np.minimum(dummy_1min_df['low'], dummy_1min_df[['open','close']].min(axis=1))
    dummy_1min_df['high'] = np.maximum(dummy_1min_df['high'], dummy_1min_df[['open','close']].max(axis=1))


    print("\nTesting calculate_volume_delta_zscore...")
    z_score_output_df = calculate_volume_delta_zscore(dummy_1min_df.copy(), z_len=5) # Use smaller z_len for test data
    print("Z-Score DataFrame head:\n", z_score_output_df.head())
    print("Z-Score DataFrame tail:\n", z_score_output_df.tail())
    if not z_score_output_df.empty:
        print("Z-Score NaNs:\n", z_score_output_df.isnull().sum())

    # Prepare 30-min data for VWAP and ATR tests from the Z-score output (which includes 30min close)
    # or by resampling the original 1-min data again.
    # For simplicity, let's use the 'close' from z_score_output_df and merge full OHLCV if needed.
    # However, calculate_session_vwap_bands expects full OHLCV.
    
    df_30m_for_vwap_atr = dummy_1min_df.resample('30min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna()

    if not df_30m_for_vwap_atr.empty:
        print("\nTesting calculate_session_vwap_bands...")
        vwap_output_df = calculate_session_vwap_bands(df_30m_for_vwap_atr.copy())
        print("VWAP DataFrame head:\n", vwap_output_df.head())
        if not vwap_output_df.empty:
            print("VWAP NaNs:\n", vwap_output_df.isnull().sum())

        print("\nTesting calculate_atr...")
        atr_output_series = calculate_atr(df_30m_for_vwap_atr.copy(), period=5) # Smaller period for test
        print("ATR Series head:\n", atr_output_series.head())
        if not atr_output_series.empty:
            print("ATR NaNs:", atr_output_series.isnull().sum())
    else:
        print("Not enough 30-min data to test VWAP/ATR.")
