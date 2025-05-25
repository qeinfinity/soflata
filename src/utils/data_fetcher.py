import pandas as pd
import os
from datetime import datetime, timezone
from binance.client import Client # From python-binance library

# Define a directory for caching data (assuming this is already defined as before)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
if not os.path.exists(DATA_CACHE_DIR):
    os.makedirs(DATA_CACHE_DIR)
    # print(f"[DataFetcher] Created cache directory: {DATA_CACHE_DIR}") # Already printed if new

def get_binance_client():
    """Initializes and returns a Binance client instance."""
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    return Client(api_key, api_secret)

def generate_klines_filepath(symbol, interval, start_dt, end_dt):
    """Generates a standardized filepath for kline data."""
    filename = f"{symbol.upper()}_{interval}_binance_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}.csv"
    return os.path.join(DATA_CACHE_DIR, filename)

def fetch_and_cache_hist_klines(symbol, interval, start_str, end_str, force_refetch=False):
    """
    Fetches historical klines from Binance for a given symbol, interval, and date range.
    Saves data to CSV and loads from CSV if already available.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        interval (str): Kline interval (e.g., "1m", "30m", "1d"). Corresponds to Client.KLINE_INTERVAL_*.
        start_str (str): Start date string (e.g., "1 Jan, 2022").
        end_str (str): End date string (e.g., "1 Jan, 2023"). This will be passed to the generator.
        force_refetch (bool): If True, will always download from Binance, ignoring cache.

    Returns:
        pandas.DataFrame: DataFrame with kline data, or None if fetching fails.
    """
    client = get_binance_client()

    try:
        start_dt_utc = pd.to_datetime(start_str, utc=True)
        # For the filename, we want the end_str to reflect the requested period.
        # However, get_historical_klines_generator uses end_str as the "up to" point.
        # If end_str is "1 Apr, 2023", it fetches data UP TO the start of April 1st.
        # So, the data will include March 31st, 2023, 23:59.
        # The filename should ideally reflect the *inclusive* end date of the data.
        # Let's parse end_str to use for the filename, and pass it directly to the API.
        # The API handles the "up to" logic.
        end_dt_utc_for_filename = pd.to_datetime(end_str, utc=True)
    except Exception as e:
        print(f"[ERROR] Invalid date format for start/end string: {e}")
        return None

    filepath = generate_klines_filepath(symbol, interval, start_dt_utc, end_dt_utc_for_filename)

    if not force_refetch and os.path.exists(filepath):
        print(f"[DataFetcher] Loading {interval} klines for {symbol} from cache: {filepath}")
        try:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            cols_to_numeric = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades',
                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            for col in cols_to_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"[DataFetcher] Loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load cached data from {filepath}: {e}. Will attempt to refetch.")

    print(f"[DataFetcher] Fetching {interval} klines for {symbol} from {start_str} up to {end_str} from Binance...")
    
    klines_list = []
    try:
        # CORRECTED CALL: Removed 'end_str_kst', just use 'end_str' (or it's optional)
        klines_generator = client.get_historical_klines_generator(symbol, interval, start_str, end_str=end_str)
        for kline in klines_generator:
            klines_list.append(kline)
            if len(klines_list) % 50000 == 0 : 
                 print(f"[DataFetcher] Fetched {len(klines_list)} klines for {symbol}...")

    except Exception as e: 
        print(f"[ERROR] Binance API call failed: {e}")
        return None

    if not klines_list:
        print(f"[WARN] No klines returned from Binance for {symbol} {interval} from {start_str} to {end_str}.")
        return pd.DataFrame() 

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    
    df = pd.DataFrame(klines_list, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    
    print(f"[DataFetcher] Successfully fetched {len(df)} klines for {symbol}.")

    try:
        df.to_csv(filepath)
        print(f"[DataFetcher] Saved data to cache: {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save data to cache {filepath}: {e}")

    return df

if __name__ == '__main__':
    print("--- Testing Binance Data Fetcher (Corrected) ---")
    
    start_date_fixed = "1 Jan, 2023"
    # The end_str for get_historical_klines_generator is exclusive for the day part.
    # If you want data *including* April 1st, you might need to set end_str to "2 Apr, 2023".
    # Let's test fetching data for Jan 1st up to (but not including) Jan 2nd.
    # To fetch data for "1 Jan, 2023" to "1 Apr, 2023" (inclusive of Apr 1st data up to 23:59),
    # your end_str should typically be the day *after* your desired last day.
    # So if you want data UP TO the END of "1 Apr, 2023", you might specify "2 Apr, 2023" as end_str.
    # Let's adjust the main.py call slightly for clarity too.
    
    # Test with a very small range first:
    test_start = "1 Jan, 2024 00:00:00" # More precise start
    test_end = "1 Jan, 2024 01:00:00"   # Fetch one hour of data

    print(f"\nAttempting to fetch {Client.KLINE_INTERVAL_1MINUTE} data for BTCUSDT from {test_start} up to {test_end}")
    df_klines = fetch_and_cache_hist_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, test_start, test_end)

    if df_klines is not None and not df_klines.empty:
        print(f"\nFetched/Loaded DataFrame for BTCUSDT:")
        print(df_klines.head())
        print(f"\nTail:")
        print(df_klines.tail()) # Important to check the end timestamp
        print(f"\nShape: {df_klines.shape}")
        df_klines.info()
        print(f"\nNaN counts per column:\n{df_klines.isnull().sum()}")
    elif df_klines is not None and df_klines.empty:
        print(f"\nNo data returned for BTCUSDT in the specified period.")
    else:
        print(f"\nFailed to fetch/load data for BTCUSDT.")

    # Example for 30-minute data
    # interval_30m = Client.KLINE_INTERVAL_30MINUTE
    # print(f"\nAttempting to fetch {interval_30m} data for {symbol_to_fetch} from {start_date_fixed} to {end_date_fixed}")
    # df_klines_30m = fetch_and_cache_hist_klines(symbol_to_fetch, interval_30m, start_date_fixed, end_date_fixed)
    # if df_klines_30m is not None and not df_klines_30m.empty:
    #     print(df_klines_30m.head())
