# File: main.py (Showing relevant parts of run_backtest function)

import pandas as pd
from src.utils import data_fetcher
from src.core import indicators
from src.backtester.engine import BacktestEngine
from binance.client import Client # To use Client.KLINE_INTERVAL_ constants

# --- Configuration ---
SYMBOL = "BTCUSDT"
KLINE_INTERVAL_1MIN = Client.KLINE_INTERVAL_1MINUTE # "1m"
START_DATE_STR = "1 Jan, 2023"
# If you want data covering the full day of "1 Apr, 2023", set end to the next day.
END_DATE_FOR_FETCH = "2 Apr, 2023" # Data up to start of April 2nd for API
INITIAL_EQUITY = 100000

# --- Helper to ensure data is ready (assuming prepare_base_data is as before) ---
def prepare_base_data(symbol, interval, start_str, end_str_for_api_call, force_refetch=False):
    print(f"[Main] Preparing base {interval} data for {symbol} from {start_str} up to {end_str_for_api_call}...")
    df_1min = data_fetcher.fetch_and_cache_hist_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str_for_api_call,
        force_refetch=force_refetch
    )
    if df_1min is None or df_1min.empty:
        print(f"[ERROR] Failed to fetch/load 1-minute base data for {symbol}. Exiting.")
        return None
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df_1min.columns:
            print(f"[ERROR] Missing required column '{col}' in fetched 1-minute data. Exiting.")
            return None
    
    print(f"[Main] Base 1-minute data prepared successfully. Shape: {df_1min.shape}")
    return df_1min

# --- Main Backtest Orchestration ---
def run_backtest():
    print("============================================")
    print("    Initializing VolDelta-MR Backtest       ")
    print("============================================")

    df_1min_raw = prepare_base_data(SYMBOL, KLINE_INTERVAL_1MIN, START_DATE_STR, END_DATE_FOR_FETCH, force_refetch=False)
    if df_1min_raw is None:
        return

    print("\n[Main] Preprocessing data and calculating indicators...")
    
    print("[Main] Calculating Volume Delta Z-Score...")
    df_with_zscore_30min = indicators.calculate_volume_delta_zscore(df_1min_raw.copy()) 
    if df_with_zscore_30min.empty or 'volume_delta_zscore' not in df_with_zscore_30min.columns:
        print("[ERROR] Volume Delta Z-Score calculation failed or returned empty/invalid DataFrame. Exiting.")
        return

    print("[Main] Resampling raw 1-min data to 30-min OHLCV...")
    df_ohlcv_30min = df_1min_raw.resample('30min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna() 
    
    if df_ohlcv_30min.empty:
        print("[ERROR] 30-min OHLCV DataFrame is empty after resampling. Exiting.")
        return

    print("[Main] Merging Z-Score data with 30-min OHLCV...")
    df_master_30min = df_ohlcv_30min.copy()
    # Merge only the essential Z-score column (and perhaps 'sampled_roll' if needed elsewhere)
    # Assuming 'volume_delta_zscore' and 'sampled_roll' are in df_with_zscore_30min
    columns_to_merge_from_zscore = ['volume_delta_zscore']
    if 'sampled_roll' in df_with_zscore_30min.columns: # Optional, if you use sampled_roll later
        columns_to_merge_from_zscore.append('sampled_roll')
    
    df_master_30min = pd.merge(df_master_30min, 
                               df_with_zscore_30min[columns_to_merge_from_zscore],
                               left_index=True, right_index=True, how='inner')

    print("[Main] Calculating Session VWAP & Bands...")
    df_master_30min = indicators.calculate_session_vwap_bands(df_master_30min)

    print("[Main] Calculating ATR...")
    df_master_30min['atr_20_30m'] = indicators.calculate_atr(df_master_30min, period=20)
    median_atr_lookback = 50 
    df_master_30min['median_atr_20_30m'] = df_master_30min['atr_20_30m'].rolling(
        window=median_atr_lookback, min_periods=max(1, median_atr_lookback // 2)
    ).median()

    # --- CRITICAL: Ensure Causality for Signal Generation ---
    # Shift indicator values used for decision making by 1 bar.
    # This means for bar 't', we use indicators calculated at 't-1'.
    print("[Main] Shifting indicators for causal signal generation...")
    cols_to_shift_for_signal = [
        'volume_delta_zscore', 
        'session_vwap', 
        'session_stdev',
        'vwap_upper1', 'vwap_lower1',
        'vwap_upper2', 'vwap_lower2',
        'vwap_upper3', 'vwap_lower3',
        'atr_20_30m', 
        'median_atr_20_30m'
    ]
    
    for col in cols_to_shift_for_signal:
        if col in df_master_30min.columns:
            df_master_30min[f'{col}_lagged'] = df_master_30min[col].shift(1)
        else:
            print(f"[WARN] Column {col} not found for shifting in df_master_30min.")

    print(f"[Main] Shape before final dropna (post-indicators & shift): {df_master_30min.shape}")
    df_master_30min.dropna(inplace=True) 
    print(f"[Main] Shape after final dropna (master data ready): {df_master_30min.shape}")

    if df_master_30min.empty:
        print("[ERROR] Master 30-min DataFrame is empty after all indicator calculations and shifting. Cannot backtest.")
        return

    print("\n[Main] df_master_30min (with lagged signals) sample head:")
    print(df_master_30min.head())
    
    print(f"\n[Main] Initializing BacktestEngine with initial equity: {INITIAL_EQUITY}...")
    engine = BacktestEngine(df_master_30min, initial_equity=INITIAL_EQUITY)
    
    print("[Main] Starting backtest simulation run...")
    engine.run_simulation()
    
    print("\n============================================")
    print("    Backtest Run Concluded                  ")
    print("============================================")

if __name__ == "__main__":
    print("Welcome to VolDelta-MR System! v1.4 (Causality & Risk Updated)")
    run_backtest()
    print("\nSystem execution finished.")
