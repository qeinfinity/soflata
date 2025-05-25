# File: main.py (Modified)

import pandas as pd
from src.utils import data_fetcher # NEW: Import the data fetcher
from src.core import indicators
from src.backtester.engine import BacktestEngine
from binance.client import Client # To use Client.KLINE_INTERVAL_ constants

# --- Configuration ---
SYMBOL = "BTCUSDT"
KLINE_INTERVAL_1MIN = Client.KLINE_INTERVAL_1MINUTE # "1m"
# PRD needs 6+ months for backtesting, 12+ months for parameter optimization
# For a robust test, let's aim for 1 year of data.
# NOTE: Fetching 1 year of 1-minute data can be large and take time on first fetch.
# Start with a smaller period for initial testing, e.g., 3 months.
START_DATE_STR = "1 Jan, 2023"  # Adjust as needed
END_DATE_STR = "1 Apr, 2023"    # Adjust for desired backtest period (e.g., "1 Jan, 2024" for 1 year)
INITIAL_EQUITY = 100000

# --- Helper to ensure data is ready ---
def prepare_base_data(symbol, interval, start_str, end_str, force_refetch=False):
    print(f"[Main] Preparing base {interval} data for {symbol} from {start_str} to {end_str}...")
    df_1min = data_fetcher.fetch_and_cache_hist_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str,
        force_refetch=force_refetch
    )
    if df_1min is None or df_1min.empty:
        print(f"[ERROR] Failed to fetch/load 1-minute base data for {symbol}. Exiting.")
        return None
    
    # Ensure necessary columns exist and are numeric for indicator calculations
    # data_fetcher already handles basic numeric conversion and timestamp index
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

    # 1. Load/Fetch Historical 1-Minute Data
    df_1min_raw = prepare_base_data(SYMBOL, KLINE_INTERVAL_1MIN, START_DATE_STR, END_DATE_STR, force_refetch=False)
    if df_1min_raw is None:
        return

    # 2. Preprocess Data & Calculate Indicators to create df_master_30min
    print("\n[Main] Preprocessing data and calculating indicators...")
    
    # Ensure 'hlc3' is calculated on the 1-min data *before* Z-score calculation
    # The calculate_volume_delta_zscore function in indicators.py already does this.

    # Calculate Z-Score (this function resamples 1-min to 30-min and adds 'volume_delta_zscore')
    # It returns a 30-min DF.
    print("[Main] Calculating Volume Delta Z-Score...")
    # Pass a copy to avoid modifying the raw 1-min data if needed elsewhere
    df_with_zscore_30min = indicators.calculate_volume_delta_zscore(df_1min_raw.copy()) 
    if df_with_zscore_30min.empty or 'volume_delta_zscore' not in df_with_zscore_30min.columns:
        print("[ERROR] Volume Delta Z-Score calculation failed or returned empty/invalid DataFrame. Exiting.")
        return

    # We need full 30-min OHLCV data for VWAP and ATR. Resample the raw 1-min data.
    print("[Main] Resampling raw 1-min data to 30-min OHLCV...")
    df_ohlcv_30min = df_1min_raw.resample('30min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna() # DropNA for periods where there might be no 1-min data (e.g. exchange downtime)
    
    if df_ohlcv_30min.empty:
        print("[ERROR] 30-min OHLCV DataFrame is empty after resampling. Exiting.")
        return

    # Merge Z-score results with the full 30-min OHLCV.
    # Both should have a 30-min DatetimeIndex.
    print("[Main] Merging Z-Score data with 30-min OHLCV...")
    # We only need 'volume_delta_zscore' and 'sampled_roll' (if used directly) from df_with_zscore_30min.
    # The rest (EMAs, hist) were intermediate.
    # The current calculate_volume_delta_zscore returns the full 30min df with intermediate steps,
    # so we can merge on index and just pick the columns we need.
    # For robustness, let's pick specific columns from z-score df for merge if it has many.
    # Assuming df_with_zscore_30min index is the 30T timestamp.
    
    # Let's refine df_master_30min creation to ensure it's built from df_ohlcv_30min
    df_master_30min = df_ohlcv_30min.copy()
    
    # Add Z-score to df_master_30min
    # The calculate_volume_delta_zscore from previous turn returns a df with 'volume_delta_zscore'
    # and other intermediate columns like 'sampled_roll', 'close' (30min).
    # We merge it based on index.
    df_master_30min = pd.merge(df_master_30min, 
                               df_with_zscore_30min[['volume_delta_zscore', 'sampled_roll']], # Add 'sampled_roll' if it's used elsewhere
                               left_index=True, right_index=True, how='inner')


    print("[Main] Calculating Session VWAP & Bands...")
    df_master_30min = indicators.calculate_session_vwap_bands(df_master_30min) # Pass the 30-min DF

    print("[Main] Calculating ATR...")
    df_master_30min['atr_20_30m'] = indicators.calculate_atr(df_master_30min, period=20)
    # Calculate median ATR for volatility adjustment in position sizing
    # PRD suggests median_atr20 / current_atr20. Let's use a 50-bar median for stability.
    median_atr_lookback = 50 
    df_master_30min['median_atr_20_30m'] = df_master_30min['atr_20_30m'].rolling(window=median_atr_lookback, min_periods=median_atr_lookback // 2).median()

    # Drop any initial NaNs created by rolling indicators (Z-score, VWAP StDev, ATR, Median ATR)
    print(f"[Main] Shape before final dropna: {df_master_30min.shape}")
    df_master_30min.dropna(inplace=True)
    print(f"[Main] Shape after final dropna (master data ready): {df_master_30min.shape}")

    if df_master_30min.empty:
        print("[ERROR] Master 30-min DataFrame is empty after all indicator calculations. Cannot backtest.")
        return

    print("\n[Main] df_master_30min sample head:")
    print(df_master_30min.head())
    print("\n[Main] df_master_30min sample tail:")
    print(df_master_30min.tail())


    # 3. Initialize and Run Backtesting Engine
    print(f"\n[Main] Initializing BacktestEngine with initial equity: {INITIAL_EQUITY}...")
    engine = BacktestEngine(df_master_30min, initial_equity=INITIAL_EQUITY)
    
    print("[Main] Starting backtest simulation run...")
    engine.run_simulation()
    
    # 4. Generate Performance Report (done by engine.run_simulation() -> engine.generate_report())
    print("\n============================================")
    print("    Backtest Run Concluded                  ")
    print("============================================")


if __name__ == "__main__":
    print("Welcome to VolDelta-MR System! v1.4")
    run_backtest()
    print("\nSystem execution finished.")

