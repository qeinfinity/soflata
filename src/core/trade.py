# VolDelta-MR: src/core/trade.py
# Optional: Class or data structures for managing active trade state.

class Trade:
    def __init__(self, entry_timestamp, direction, entry_price, initial_vwap0, initial_sigma0):
        self.active = True
        self.entry_timestamp = entry_timestamp
        self.direction = direction
        self.entry_price = entry_price
        self.initial_vwap0 = initial_vwap0
        self.initial_sigma0 = initial_sigma0
        
        self.t1_entry_size = 0.0
        self.t1_current_size = 0.0
        self.t1_stop_price = 0.0
        self.t1_status = "PENDING" # PENDING, OPEN, CLOSED

        self.t2_entry_size = 0.0
        self.t2_current_size = 0.0
        self.t2_stop_price = 0.0
        self.t2_status = "PENDING" # PENDING, OPEN, PARTIAL_EXIT_1, CLOSED
        
        self.bars_held = 0
        self.log = [] # Log of actions for this trade
        self.pnl = 0.0

        print(f"Trade object initialized for {direction} @ {entry_price} on {entry_timestamp}")

    def add_log(self, message):
        self.log.append(message)
        print(f"[Trade {self.entry_timestamp}] {message}")

    # Add more methods for managing state, P&L, etc.

def create_new_trade_state_dict(entry_timestamp, direction, 
                                entry_price_at_band, # NEW: Price of the band that was hit
                                actual_entry_price,    # NEW: Price after slippage
                                vwap0_lagged, sigma0_lagged):
    return {
        'active': True,
        'entry_timestamp': entry_timestamp,
        'direction': direction,
        'entry_price_at_band': entry_price_at_band, # Store the band price
        'actual_entry_price': actual_entry_price,   # Store price after slippage (used for P&L)
        'initial_vwap0': vwap0_lagged, # VWAP at start of entry bar (from t-1)
        'initial_sigma0': sigma0_lagged, # Sigma at start of entry bar (from t-1)
        't1_entry_size': 0.0, 't1_current_size': 0.0, 't1_stop_price': 0.0, 't1_status': 'PENDING',
        't2_entry_size': 0.0, 't2_current_size': 0.0, 't2_stop_price': 0.0, 't2_status': 'PENDING',
        'bars_held': 0,
        'log': [],
        'pnl': 0.0,
        'initial_total_risk_dollars': 0.0, # Will be populated by BacktestEngine
        'scaled_out_t2_50_pct': False,
        'scaled_out_t2_next_25_pct': False,
        'vwap_1sigma_target_hit_for_t1': False 
    }

# Example: Simple trade dictionary structure
# current_trade = {
#     'active': True,
#     'entry_timestamp': pd.Timestamp(...),
#     'entry_bar_index': ..., # index in df_master_30min
#     'direction': 'SHORT' or 'LONG',
#     'entry_price': 12345.67,
#     'initial_vwap0': vwap_val_at_entry, # VWAP at entry bar
#     'initial_sigma0': sigma_val_at_entry, # Sigma at entry bar
#     't1_entry_size': units_of_btc_t1,
#     't1_current_size': units_of_btc_t1, # Starts same as entry_size
#     't1_stop_price': initial_stop_t1,
#     't1_status': 'OPEN', # 'OPEN', 'CLOSED'
#     't2_entry_size': units_of_btc_t2,
#     't2_current_size': units_of_btc_t2,
#     't2_stop_price': initial_stop_t2,
#     't2_status': 'OPEN', # 'OPEN', 'PARTIAL_EXIT_1', 'CLOSED'
#     # Add more fields as needed for P&L tracking, exit reasons, etc.
# }
# no_active_trade = {'active': False}

if __name__ == '__main__':
    print("Testing Trade class/dict (placeholders)...")
    # trade_obj = Trade(pd.Timestamp.now(), "SHORT", 60000, 59000, 100)
    # trade_obj.add_log("Trade opened.")
    trade_dict = create_new_trade_state_dict(pd.Timestamp.now(tz='UTC'), "LONG", 50000, 51000, 150)
    print(trade_dict)

