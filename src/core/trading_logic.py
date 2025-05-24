# VolDelta-MR: src/core/trading_logic.py
# Contains functions for entry signals, position sizing, stop management, and exits.

import pandas as pd

# Placeholder for check_entry_signal
# In src/core/trading_logic.py
def check_entry_signal(bar_data):
    """
    Checks entry conditions for a single 30-minute bar.
    Assumes bar_data is a pandas Series (a row from df_master_30min)
    containing all necessary pre-calculated indicators for that bar's close
    (e.g., vwap_upper2, vwap_lower2, volume_delta_zscore).
    """
    signal = None
    entry_price = None

    # Short Entry Condition
    # Price touches VWAP + 2σ: bar_data['high'] >= bar_data['vwap_upper2']
    # Z-Score condition: bar_data['volume_delta_zscore'] >= 1.0
    if bar_data['high'] >= bar_data['vwap_upper2'] and bar_data['volume_delta_zscore'] >= 1.0:
        signal = 'SHORT'
        # Assume entry at the band if touched. For simplicity in backtest.
        # More advanced: model entry somewhere between band and high/low.
        entry_price = bar_data['vwap_upper2'] 
    
    # Long Entry Condition (can't be both short and long on same bar logic)
    # Price touches VWAP - 2σ: bar_data['low'] <= bar_data['vwap_lower2']
    # Z-Score condition: bar_data['volume_delta_zscore'] <= -1.0
    elif bar_data['low'] <= bar_data['vwap_lower2'] and bar_data['volume_delta_zscore'] <= -1.0:
        signal = 'LONG'
        entry_price = bar_data['vwap_lower2']
        
    # PRD 3.1 Optional ADX filter could be added here if bar_data contains ADX value.
    # e.g., if signal == 'SHORT' and bar_data['adx'] > SOME_THRESHOLD and bar_data['plus_di'] > bar_data['minus_di']: signal = None

    return signal, entry_price


# Placeholder for calculate_initial_stop_prices
def calculate_initial_stop_prices(entry_price, vwap_at_entry, sigma_at_entry, trade_direction):
    """
    Calculates initial stop prices for T1 and T2.
    sigma_at_entry is the session_stdev value at the time of entry.
    """
    if trade_direction == 'SHORT':
        stop_t1 = entry_price + (0.25 * sigma_at_entry)
        stop_t2 = vwap_at_entry + (3.00 * sigma_at_entry)
    elif trade_direction == 'LONG':
        stop_t1 = entry_price - (0.25 * sigma_at_entry)
        stop_t2 = vwap_at_entry - (3.00 * sigma_at_entry)
    else:
        raise ValueError("Invalid trade_direction")
    return stop_t1, stop_t2

# Placeholder for calculate_position_size_for_tranche
def calculate_position_size_for_tranche(account_equity, tranche_risk_fraction, 
                                        entry_price, stop_price, 
                                        current_atr, median_atr, # For ATR volatility adjustment
                                        target_amp_pct=None, price_amp_24h=None, # For CoinGlass method (optional)
                                        vol_adj_method='ATR'): # 'ATR', 'COINGLASS', or 'NONE'
    """
    Calculates position size for a single tranche.
    tranche_risk_fraction is e.g., 0.002 for T1 (0.2%).
    """
    risk_budget_dollars = account_equity * tranche_risk_fraction
    stop_distance_dollars = abs(entry_price - stop_price)

    if stop_distance_dollars == 0: # Avoid division by zero
        return 0 
        
    raw_size_units = risk_budget_dollars / stop_distance_dollars # e.g., in BTC

    vol_adj = 1.0
    if vol_adj_method == 'ATR':
        if current_atr > 0 and median_atr > 0 : # Avoid division by zero and ensure valid ATRs
             vol_adj = median_atr / current_atr
        # Cap vol_adj to avoid extreme sizing, e.g., vol_adj = min(max(vol_adj, 0.5), 2.0)
    elif vol_adj_method == 'COINGLASS':
        if target_amp_pct is not None and price_amp_24h is not None and price_amp_24h > 0:
            vol_adj = target_amp_pct / price_amp_24h
        # Add capping for CoinGlass vol_adj too
    
    # Apply a cap/floor to vol_adj, e.g.,
    vol_adj = min(max(vol_adj, 0.5), 1.5) # Example: Don't more than halve or 1.5x size

    position_size_units = raw_size_units * vol_adj
    
    # PRD 7.1 Position Size Limit (min of 5% 24h vol, |VolDelta 'hist'|)
    # This part is complex as 'hist' isn't directly notional.
    # Simpler for Phase 1: Max size based on % of 24h volume.
    # This limit should be applied *after* calculating total desired size for T1+T2.
    # For now, this function returns the tranche-specific calculated size.
    # The overall limit application would be in the main trading loop.

    return position_size_units


# Placeholder for update_trade_status_per_bar
def update_trade_status_per_bar(current_trade, bar_data, account_equity_tracker): # bar_data is current 30-min bar from df_master_30min
    if not current_trade['active']:
        return current_trade, [] # No active trade, no actions

    trade_actions = [] # To log what happened: ['T1_STOPPED', 'T2_SCALED_OUT_1', etc.]
    
    # --- 0. Unpack current trade state and current bar data ---
    direction = current_trade['direction']
    # ... (unpack other trade details: entry_price, initial_sigma0, t1_size, t1_stop, etc.)
    # ... (unpack bar_data: open, high, low, close, current_vwap, current_sigma, current_zscore, atr_20_30m)
    
    # --- I. Check for Stop-Loss Hits (using bar's high/low against stop prices) ---
    # Important: Check stops *before* scale-outs in case of large gap.
    # For T1:
    if current_trade['t1_status'] == 'OPEN':
        if direction == 'SHORT' and bar_data['high'] >= current_trade['t1_stop_price']:
            # T1 STOPPED OUT
            trade_actions.append(f"T1_STOPPED_OUT @ {current_trade['t1_stop_price']}")
            # Record P&L for T1, update current_trade['t1_current_size'] = 0, current_trade['t1_status'] = 'CLOSED'
            # account_equity_tracker would be updated here.
        elif direction == 'LONG' and bar_data['low'] <= current_trade['t1_stop_price']:
            # T1 STOPPED OUT
            trade_actions.append(f"T1_STOPPED_OUT @ {current_trade['t1_stop_price']}")
            # ... record P&L, update trade state ...
    # For T2: (similar logic for t2_stop_price)
    if current_trade['t2_status'] != 'CLOSED': # Check if T2 is active at all
         if direction == 'SHORT' and bar_data['high'] >= current_trade['t2_stop_price']:
            # T2 STOPPED OUT
            trade_actions.append(f"T2_STOPPED_OUT @ {current_trade['t2_stop_price']}")
            # ... record P&L, update trade state ...
         elif direction == 'LONG' and bar_data['low'] <= current_trade['t2_stop_price']:
            # T2 STOPPED OUT
            trade_actions.append(f"T2_STOPPED_OUT @ {current_trade['t2_stop_price']}")
            # ... record P&L, update trade state ...

    # If all parts of trade closed by stops, mark trade inactive and return
    if current_trade['t1_current_size'] == 0 and current_trade['t2_current_size'] == 0:
        current_trade['active'] = False
        return current_trade, trade_actions

    # --- II. Check Invalidation Rule (PRD 3.3) ---
    # Only if T1 or T2 is still somewhat open and VWAP +/- 1 sigma not yet hit for T1 (need to track this state)
    # This needs careful state: "before price reaches current VWAP±1σ"
    # Let's assume a flag 'vwap_1sigma_target_hit_for_t1' in current_trade, default False.
    if not current_trade.get('vwap_1sigma_target_hit_for_t1', False): # If 1-sigma target not hit yet
        z_flipped = False
        if direction == 'SHORT' and bar_data['volume_delta_zscore'] < 0: # Was >= +1.0, now < 0
            z_flipped = True
        elif direction == 'LONG' and bar_data['volume_delta_zscore'] > 0: # Was <= -1.0, now > 0
            z_flipped = True
        
        if z_flipped:
            invalidation_stop_price = 0
            if direction == 'SHORT':
                invalidation_stop_price = current_trade['entry_price'] + (0.1 * current_trade['initial_sigma0'])
            else: # LONG
                invalidation_stop_price = current_trade['entry_price'] - (0.1 * current_trade['initial_sigma0'])
            
            trade_actions.append(f"INVALIDATION_Z_FLIP_STOPS_MOVED_TO_{invalidation_stop_price}")
            # Update T1 stop (if T1 still open)
            if current_trade['t1_status'] == 'OPEN':
                if direction == 'SHORT':
                    current_trade['t1_stop_price'] = max(current_trade['t1_stop_price'], invalidation_stop_price) # Never widen, but invalidation is usually tighter
                else: # LONG
                    current_trade['t1_stop_price'] = min(current_trade['t1_stop_price'], invalidation_stop_price)
            # Update T2 stop
            if current_trade['t2_status'] != 'CLOSED':
                if direction == 'SHORT':
                    current_trade['t2_stop_price'] = max(current_trade['t2_stop_price'], invalidation_stop_price)
                else: # LONG
                    current_trade['t2_stop_price'] = min(current_trade['t2_stop_price'], invalidation_stop_price)
            # Re-check stops immediately with these new prices for current bar's H/L (could be complex, or apply for next bar)
            # For simplicity in backtest: new stops apply from next bar *unless* current bar's H/L already hit them.

    # --- III. Check Scale-Out & Profit-Taking Exits (PRD 3.4) ---
    # Using bar's high/low for touches, VWAP values from bar_data (which are current session VWAP)
    current_vwap = bar_data['session_vwap']
    current_sigma = bar_data['session_stdev'] # This is sigma_t

    # Target 1: Price reaches current VWAP ± 1σ
    target_1sigma_price = 0
    if direction == 'SHORT': target_1sigma_price = current_vwap - (1 * current_sigma)
    else: target_1sigma_price = current_vwap + (1 * current_sigma)

    scaled_out_t2_50_pct = False # Flag to ensure this happens only once per trade
    if not current_trade.get('scaled_out_t2_50_pct', False) and current_trade['t2_current_size'] > 0:
        if (direction == 'SHORT' and bar_data['low'] <= target_1sigma_price) or \
           (direction == 'LONG' and bar_data['high'] >= target_1sigma_price):
            
            amount_to_close_t2 = current_trade['t2_entry_size'] * 0.50 # 50% of *original* T2
            if current_trade['t2_current_size'] >= amount_to_close_t2 : # Ensure we have enough to close
                trade_actions.append(f"T2_SCALED_OUT_50% @ {target_1sigma_price}")
                # Record P&L for this portion of T2, update current_trade['t2_current_size']
                current_trade['t2_current_size'] -= amount_to_close_t2
                current_trade['scaled_out_t2_50_pct'] = True
                current_trade['vwap_1sigma_target_hit_for_t1'] = True # Mark T1's first objective conceptually hit

                # Trail REMAINING T2 to VWAP + 1.5σ (short) / VWAP - 1.5σ (long)
                new_t2_stop = 0
                if direction == 'SHORT': new_t2_stop = current_vwap + (1.5 * current_sigma)
                else: new_t2_stop = current_vwap - (1.5 * current_sigma)
                
                # Update T2 stop (never widen from its *current* protective position after this scale out)
                if direction == 'SHORT': current_trade['t2_stop_price'] = max(current_trade['t2_stop_price'], new_t2_stop) 
                else: current_trade['t2_stop_price'] = min(current_trade['t2_stop_price'], new_t2_stop)
                trade_actions.append(f"T2_STOP_TRAILED_POST_SCALE1_TO_{current_trade['t2_stop_price']}")


    # Target 2: Price touches current VWAP
    target_vwap_price = current_vwap
    if (direction == 'SHORT' and bar_data['low'] <= target_vwap_price) or \
       (direction == 'LONG' and bar_data['high'] >= target_vwap_price):
        
        # Close all remaining T1
        if current_trade['t1_status'] == 'OPEN' and current_trade['t1_current_size'] > 0:
            trade_actions.append(f"T1_CLOSED_AT_VWAP @ {target_vwap_price}")
            # Record P&L for T1, current_trade['t1_current_size'] = 0, current_trade['t1_status'] = 'CLOSED'
            current_trade['t1_current_size'] = 0
            current_trade['t1_status'] = 'CLOSED'
            current_trade['vwap_1sigma_target_hit_for_t1'] = True # Ensure marked

        # Close a further 25% of T2 (original size)
        if not current_trade.get('scaled_out_t2_next_25_pct', False) and current_trade['t2_current_size'] > 0:
            amount_to_close_t2_further = current_trade['t2_entry_size'] * 0.25
            actual_close_amount = min(amount_to_close_t2_further, current_trade['t2_current_size']) # Close what's left if less than 25%
            if actual_close_amount > 0:
                trade_actions.append(f"T2_SCALED_OUT_FURTHER_25% @ {target_vwap_price}")
                # Record P&L, update current_trade['t2_current_size']
                current_trade['t2_current_size'] -= actual_close_amount
                current_trade['scaled_out_t2_next_25_pct'] = True
    
    # Target 3: Price overshoots VWAP by –0.5σ (short) / +0.5σ (long)
    # This applies to the *last* T2 piece.
    if current_trade['t2_current_size'] > 0: # Only if there's T2 left
        overshoot_target_price = 0
        new_trail_stop_for_residual_t2 = 0
        
        if direction == 'SHORT': 
            overshoot_target_price = current_vwap - (0.5 * current_sigma)
            if bar_data['low'] <= overshoot_target_price:
                # Trail last T2 with max(VWAP, price + 1σ) -> for SHORT, this means stop moves DOWN
                # price here would be the overshoot_target_price or bar_data['low']
                stop_anchor1 = current_vwap 
                stop_anchor2 = bar_data['close'] + (1 * current_sigma) # Using bar's close as "price"
                new_trail_stop_for_residual_t2 = max(stop_anchor1, stop_anchor2) # Stop cannot be above current VWAP, but can be price+1sigma
                current_trade['t2_stop_price'] = max(current_trade['t2_stop_price'], new_trail_stop_for_residual_t2) # Stop moves down for short
                trade_actions.append(f"T2_RESIDUAL_TRAILED_OVERSHOOT_TO_{current_trade['t2_stop_price']}")
        
        else: # LONG
            overshoot_target_price = current_vwap + (0.5 * current_sigma)
            if bar_data['high'] >= overshoot_target_price:
                # Trail last T2 with min(VWAP, price - 1σ) -> for LONG, stop moves UP
                stop_anchor1 = current_vwap
                stop_anchor2 = bar_data['close'] - (1 * current_sigma) # Using bar's close as "price"
                new_trail_stop_for_residual_t2 = min(stop_anchor1, stop_anchor2) # Stop cannot be below current VWAP
                current_trade['t2_stop_price'] = min(current_trade['t2_stop_price'], new_trail_stop_for_residual_t2) # Stop moves up for long
                trade_actions.append(f"T2_RESIDUAL_TRAILED_OVERSHOOT_TO_{current_trade['t2_stop_price']}")


    # --- IV. Dynamic Trailing Stop Logic (PRD 3.3 - Anchor update, Never widen, Buffer floor) ---
    # This applies at the END of the bar processing for any open portions.
    # For T1 (if still open):
    if current_trade['t1_status'] == 'OPEN':
        new_t1_anchor_stop = 0
        if direction == 'SHORT': new_t1_anchor_stop = current_trade['entry_price'] + (0.25 * current_sigma) # sigma_t
        else: new_t1_anchor_stop = current_trade['entry_price'] - (0.25 * current_sigma)

        # Never widen: Compare new_anchor_stop with current_trade['t1_stop_price']
        potential_t1_stop = 0
        if direction == 'SHORT': potential_t1_stop = min(current_trade['t1_stop_price'], new_t1_anchor_stop)
        else: potential_t1_stop = max(current_trade['t1_stop_price'], new_t1_anchor_stop)
        
        # Buffer floor: stop_dist < 0.25 * ATR(20,30m) from *current market price* (bar_data['close'])
        min_stop_dist_t1 = 0.25 * bar_data['atr_20_30m']
        final_t1_stop = potential_t1_stop
        if direction == 'SHORT': # Stop is above price
            if (potential_t1_stop - bar_data['close']) < min_stop_dist_t1 :
                final_t1_stop = bar_data['close'] + min_stop_dist_t1
                # Ensure this buffer adjustment doesn't widen it from its previous position
                final_t1_stop = min(final_t1_stop, current_trade['t1_stop_price']) 
        else: # LONG, stop is below price
            if (bar_data['close'] - potential_t1_stop) < min_stop_dist_t1:
                final_t1_stop = bar_data['close'] - min_stop_dist_t1
                # Ensure this buffer adjustment doesn't widen it
                final_t1_stop = max(final_t1_stop, current_trade['t1_stop_price'])

        if final_t1_stop != current_trade['t1_stop_price']:
             trade_actions.append(f"T1_STOP_TRAILED_TO_{final_t1_stop}")
             current_trade['t1_stop_price'] = final_t1_stop

    # For T2 (if still open and not subject to a more specific trailing rule above like post-scaleout):
    if current_trade['t2_current_size'] > 0 and not current_trade.get('scaled_out_t2_50_pct', False): # If T2 not yet scaled, use its main trailing logic
        new_t2_anchor_stop = 0
        if direction == 'SHORT': new_t2_anchor_stop = current_vwap + (3.0 * current_sigma) # VWAP_t + 3*sigma_t
        else: new_t2_anchor_stop = current_vwap - (3.0 * current_sigma)

        potential_t2_stop = 0
        if direction == 'SHORT': potential_t2_stop = min(current_trade['t2_stop_price'], new_t2_anchor_stop)
        else: potential_t2_stop = max(current_trade['t2_stop_price'], new_t2_anchor_stop)

        min_stop_dist_t2 = 0.25 * bar_data['atr_20_30m']
        final_t2_stop = potential_t2_stop
        # Apply buffer floor logic similar to T1 for final_t2_stop
        if direction == 'SHORT':
            if (potential_t2_stop - bar_data['close']) < min_stop_dist_t2 :
                final_t2_stop = bar_data['close'] + min_stop_dist_t2
                final_t2_stop = min(final_t2_stop, current_trade['t2_stop_price'])
        else: # LONG
            if (bar_data['close'] - potential_t2_stop) < min_stop_dist_t2:
                final_t2_stop = bar_data['close'] - min_stop_dist_t2
                final_t2_stop = max(final_t2_stop, current_trade['t2_stop_price'])
        
        if final_t2_stop != current_trade['t2_stop_price']:
            trade_actions.append(f"T2_STOP_TRAILED_TO_{final_t2_stop}")
            current_trade['t2_stop_price'] = final_t2_stop
    
    # --- V. Check Time-Out Exit (PRD 3.4) ---
    bars_since_entry = bar_data.name - current_trade['entry_bar_index'] # .name is often the index if iterating df.iterrows()
                                                                        # Or pass bar_index explicitly
    # Assuming entry_bar_index is the integer index of the entry bar.
    # If bar_data.name is a timestamp, you'd compare timestamps.
    # A simpler way: add 'bars_held' to current_trade and increment it.
    current_trade['bars_held'] = current_trade.get('bars_held', 0) + 1
    if current_trade['bars_held'] >= 16: # 8 hours = 16 x 30-min bars
        if current_trade['t1_current_size'] > 0:
            trade_actions.append(f"T1_TIMEOUT_EXIT @ {bar_data['close']}")
            # Record P&L, update current_trade['t1_current_size'] = 0, current_trade['t1_status'] = 'CLOSED'
            current_trade['t1_current_size'] = 0
            current_trade['t1_status'] = 'CLOSED'
        if current_trade['t2_current_size'] > 0:
            trade_actions.append(f"T2_TIMEOUT_EXIT @ {bar_data['close']}")
            # Record P&L, update current_trade['t2_current_size'] = 0
            current_trade['t2_current_size'] = 0
        current_trade['active'] = False # Close whole trade on timeout
        return current_trade, trade_actions

    # Final check if trade is now fully closed
    if current_trade['t1_current_size'] == 0 and current_trade['t2_current_size'] == 0:
        current_trade['active'] = False
        
    return current_trade, trade_actions


if __name__ == '__main__':
    print("Testing trading logic functions (placeholders)...")
    # Add simple test calls here if desired
