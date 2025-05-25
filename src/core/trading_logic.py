# File: src/core/trading_logic.py

import pandas as pd
import numpy as np
import math # For math.isclose

# --- Constants (Consider moving to a dedicated config file/module later) ---
DEFAULT_FALLBACK_SPREAD_PERCENTAGE = 0.05 
SLIPPAGE_FIXED_PERCENTAGE = 0.02
RISK_PER_TRADE_T1_FRACTION = 0.002 # 0.2%
RISK_PER_TRADE_T2_FRACTION = 0.003 # 0.3%
VOL_ADJ_METHOD = 'ATR' # Options: 'ATR', 'COINGLASS', 'NONE'
VOL_ADJ_MIN_CAP = 0.5
VOL_ADJ_MAX_CAP = 1.5
TIMEOUT_BARS = 16 # 8 hours = 16 * 30-min bars (PRD: "Time-out 8 hours (16 x 30m bars)")
# Entry Z-Score thresholds
Z_SCORE_THRESHOLD_SHORT = 1.0
Z_SCORE_THRESHOLD_LONG = -1.0
# Invalidation Z-Score flip stop offset
INVALIDATION_STOP_SIGMA_OFFSET = 0.1
# T1 Probe Stop Offset
T1_PROBE_STOP_SIGMA_OFFSET = 0.25
# T2 Core Stop Offset (from VWAP at entry)
T2_CORE_STOP_SIGMA_OFFSET = 3.0
# Scale-out 1 target (VWAP +/- X sigma)
SCALE_OUT_1_SIGMA_TARGET = 1.0
# Scale-out 1 T2 trail (VWAP +/- X sigma)
SCALE_OUT_1_T2_TRAIL_SIGMA = 1.5
# Overshoot target (VWAP +/- X sigma)
OVERSHOOT_SIGMA_TARGET = 0.5 # e.g. -0.5sigma for short
OVERSHOOT_TRAIL_SIGMA = 1.0
# ATR Buffer Floor Multiplier
ATR_BUFFER_MULTIPLIER = 0.25


def calculate_slippage_amount_per_unit(execution_price, 
                                       spread_percentage=DEFAULT_FALLBACK_SPREAD_PERCENTAGE, 
                                       fixed_percentage=SLIPPAGE_FIXED_PERCENTAGE):
    """Calculates slippage amount per unit of asset based on PRD 3.6."""
    # Ensure float division
    slip_fixed = (fixed_percentage / 100.0) * execution_price 
    slip_spread_component = 0.5 * (spread_percentage / 100.0) * execution_price
    return max(slip_fixed, slip_spread_component)

def check_entry_signal(bar_data): # bar_data is current bar's OHLC + LAGGED indicators
    """
    Checks entry conditions for a single 30-minute bar using LAGGED indicators.
    bar_data['high'] and bar_data['low'] are from the current bar (for touch check).
    """
    signal = None
    entry_price_candidate = None # The price at the band that was touched

    # Use LAGGED VWAP bands and Z-score for decision
    # These values were known at the START of the current bar (end of previous bar)
    vwap_upper2_signal = bar_data.get('vwap_upper2_lagged', float('inf')) # Default if column missing
    vwap_lower2_signal = bar_data.get('vwap_lower2_lagged', float('-inf'))
    z_score_signal = bar_data.get('volume_delta_zscore_lagged', 0.0) # Default if column missing

    # Short Entry: Current bar's high touches or exceeds previous bar's VWAP+2sigma band
    if bar_data['high'] >= vwap_upper2_signal and z_score_signal >= Z_SCORE_THRESHOLD_SHORT:
        signal = 'SHORT'
        entry_price_candidate = vwap_upper2_signal # Assume entry at the band value
    
    # Long Entry: Current bar's low touches or falls below previous bar's VWAP-2sigma band
    elif bar_data['low'] <= vwap_lower2_signal and z_score_signal <= Z_SCORE_THRESHOLD_LONG:
        signal = 'LONG'
        entry_price_candidate = vwap_lower2_signal # Assume entry at the band value
        
    # TODO: Implement ADX optional filter using 'adx_lagged' if available
    # if signal and 'adx_lagged' in bar_data:
    #     if bar_data['adx_lagged'] > SOME_ADX_THRESHOLD and ... :
    #         signal = None # Filter out trade
            
    return signal, entry_price_candidate

def calculate_initial_stop_prices(entry_price_after_slippage, 
                                  vwap_at_entry_lagged,  # VWAP value from previous bar's close
                                  sigma_at_entry_lagged, # Sigma value from previous bar's close
                                  trade_direction):
    """Calculates initial stop prices for T1 and T2 using lagged VWAP/sigma values."""
    if trade_direction == 'SHORT':
        stop_t1 = entry_price_after_slippage + (T1_PROBE_STOP_SIGMA_OFFSET * sigma_at_entry_lagged)
        stop_t2 = vwap_at_entry_lagged + (T2_CORE_STOP_SIGMA_OFFSET * sigma_at_entry_lagged)
    elif trade_direction == 'LONG':
        stop_t1 = entry_price_after_slippage - (T1_PROBE_STOP_SIGMA_OFFSET * sigma_at_entry_lagged)
        stop_t2 = vwap_at_entry_lagged - (T2_CORE_STOP_SIGMA_OFFSET * sigma_at_entry_lagged)
    else:
        raise ValueError("Invalid trade_direction in calculate_initial_stop_prices")
    return stop_t1, stop_t2

def calculate_position_size_for_tranche(available_equity_for_sizing, tranche_risk_fraction, 
                                        entry_price_after_slippage, stop_price, 
                                        current_atr_lagged, median_atr_lagged,
                                        vol_adj_method=VOL_ADJ_METHOD):
    """Calculates position size using lagged ATR values."""
    risk_budget_dollars = available_equity_for_sizing * tranche_risk_fraction
    stop_distance_dollars = abs(entry_price_after_slippage - stop_price)

    if math.isclose(stop_distance_dollars, 0, abs_tol=1e-9): # Avoid division by zero
        return 0.0
            
    raw_size_units = risk_budget_dollars / stop_distance_dollars

    vol_adj = 1.0
    if vol_adj_method == 'ATR':
        # Ensure ATR values are positive and valid before division
        if current_atr_lagged > 1e-9 and median_atr_lagged > 1e-9 : 
             vol_adj = median_atr_lagged / current_atr_lagged
    # elif vol_adj_method == 'COINGLASS': # TODO: Implement CoinGlass method if desired
    #     # target_amp_pct = ... (from config)
    #     # price_amplitude_percent_24h = bar_data.get('price_amplitude_percent_24h_lagged', 0.0)
    #     # if price_amplitude_percent_24h > 1e-9:
    #     #     vol_adj = target_amp_pct / price_amplitude_percent_24h
    #     pass 
    
    # Apply min/max caps to volatility adjustment factor
    vol_adj = min(max(vol_adj, VOL_ADJ_MIN_CAP), VOL_ADJ_MAX_CAP)
    position_size_units = raw_size_units * vol_adj
    
    # TODO: Implement overall position size cap from PRD 7.1 (e.g., % of 24h volume)
    # This would be applied in the BacktestEngine *after* both tranches are sized.
    return position_size_units

def update_trade_status_per_bar(current_trade, bar_data): # bar_data has current HLOC and LAGGED indicators
    if not current_trade['active']:
        return current_trade, [], 0.0 # current_trade, actions, pnl_for_this_bar

    trade_actions = []
    pnl_for_this_bar = 0.0
    
    # Unpack trade state for convenience
    direction = current_trade['direction']
    entry_price = current_trade['entry_price'] # Actual entry price after slippage
    initial_sigma0 = current_trade['initial_sigma0'] # Sigma at time of entry (from previous bar's close)

    # Unpack current bar's actual OHLC for checking if levels were hit
    bar_high = bar_data['high']
    bar_low = bar_data['low']
    bar_close = bar_data['close']
    
    # --- Use LAGGED indicator values for decision thresholds/anchors for this bar ---
    # These were known at the START of the current bar (end of previous bar)
    decision_vwap = bar_data.get('session_vwap_lagged', current_trade['initial_vwap0']) 
    decision_sigma = bar_data.get('session_stdev_lagged', current_trade['initial_sigma0'])
    decision_zscore = bar_data.get('volume_delta_zscore_lagged', 0.0)
    decision_atr = bar_data.get('atr_20_30m_lagged', 0.0001) # Use a tiny non-zero default for ATR if missing

    # --- Helper function for closing portions of a trade ---
    def close_portion(tranche_label, amount_to_close, actual_exit_price, reason):
        nonlocal pnl_for_this_bar # Allow modification of the outer scope variable
        
        size_key_current = f"{tranche_label}_current_size"
        status_key = f"{tranche_label}_status"
        
        if current_trade.get(size_key_current, 0.0) <= 1e-9: # Effectively zero or key missing
            return

        amount_can_close = current_trade[size_key_current]
        actual_amount_closed = min(amount_to_close, amount_can_close)
        
        if actual_amount_closed <= 1e-9: return

        # Apply Slippage to the actual_exit_price
        slippage_per_unit = calculate_slippage_amount_per_unit(actual_exit_price)
        
        effective_exit_price = 0.0
        if direction == 'SHORT': # Selling to open short, so buying back to close
            effective_exit_price = actual_exit_price + slippage_per_unit # We buy back higher (worse price)
        else: # LONG, buying to open long, so selling to close
            effective_exit_price = actual_exit_price - slippage_per_unit # We sell lower (worse price)

        # Calculate P&L for this portion
        portion_pnl = 0.0
        if direction == 'SHORT':
            portion_pnl = (entry_price - effective_exit_price) * actual_amount_closed
        else: # LONG
            portion_pnl = (effective_exit_price - entry_price) * actual_amount_closed
        
        pnl_for_this_bar += portion_pnl
        current_trade['pnl'] += portion_pnl # Update total P&L for the trade object
        current_trade[size_key_current] -= actual_amount_closed
        
        action_detail = (f"{tranche_label}_{reason} @ {actual_exit_price:.2f} "
                         f"(eff: {effective_exit_price:.2f}), "
                         f"Closed: {actual_amount_closed:.8f}, PnL: {portion_pnl:.2f}")
        trade_actions.append(action_detail)
        current_trade['log'].append(f"{bar_data.name}: {action_detail}") # bar_data.name is timestamp

        if math.isclose(current_trade[size_key_current], 0.0, abs_tol=1e-9):
            current_trade[size_key_current] = 0.0
            current_trade[status_key] = 'CLOSED'
            trade_actions.append(f"{tranche_label}_FULLY_CLOSED")
            current_trade['log'].append(f"{bar_data.name}: {tranche_label}_FULLY_CLOSED")

    # --- Order of Operations as per PRD Review ---
    # I. Check for Stop-Loss Hits (using bar's H/L against current_trade stop prices)
    if current_trade['t1_status'] == 'OPEN':
        stop_price_t1 = current_trade['t1_stop_price']
        if (direction == 'SHORT' and bar_high >= stop_price_t1) or \
           (direction == 'LONG' and bar_low <= stop_price_t1):
            close_portion('t1', current_trade['t1_current_size'], stop_price_t1, "STOPPED_OUT")
            
    if current_trade['t2_status'] != 'CLOSED' and current_trade['t2_current_size'] > 1e-9 :
        stop_price_t2 = current_trade['t2_stop_price']
        if (direction == 'SHORT' and bar_high >= stop_price_t2) or \
           (direction == 'LONG' and bar_low <= stop_price_t2):
            close_portion('t2', current_trade['t2_current_size'], stop_price_t2, "STOPPED_OUT")

    # If all parts of trade closed by stops, mark trade inactive and return
    if math.isclose(current_trade.get('t1_current_size', 0.0), 0.0, abs_tol=1e-9) and \
       math.isclose(current_trade.get('t2_current_size', 0.0), 0.0, abs_tol=1e-9):
        current_trade['active'] = False
        return current_trade, trade_actions, pnl_for_this_bar

    # --- II. Check Invalidation Rule (PRD 3.3) ---
    # Uses decision_zscore (from previous bar's close)
    # Only if T1 or T2 is still somewhat open and VWAP +/- 1 sigma not yet hit (for T1's perspective)
    if not current_trade.get('vwap_1sigma_target_hit_for_t1', False) and \
       (current_trade['t1_status'] == 'OPEN' or current_trade['t2_current_size'] > 1e-9):
        
        z_flipped = (direction == 'SHORT' and decision_zscore < 0) or \
                    (direction == 'LONG' and decision_zscore > 0)
        if z_flipped:
            invalidation_stop_price = (entry_price + (INVALIDATION_STOP_SIGMA_OFFSET * initial_sigma0) if direction == 'SHORT' 
                                       else entry_price - (INVALIDATION_STOP_SIGMA_OFFSET * initial_sigma0))
            action_msg = f"INVALIDATION_Z_FLIP_STOPS_MOVED_TO_{invalidation_stop_price:.2f}"
            trade_actions.append(action_msg)
            current_trade['log'].append(f"{bar_data.name}: {action_msg}")

            # Update stops (tighten only)
            if current_trade['t1_status'] == 'OPEN':
                current_trade['t1_stop_price'] = (min(current_trade['t1_stop_price'], invalidation_stop_price) if direction == 'SHORT' 
                                                  else max(current_trade['t1_stop_price'], invalidation_stop_price))
            if current_trade['t2_current_size'] > 1e-9 and current_trade['t2_status'] != 'CLOSED':
                current_trade['t2_stop_price'] = (min(current_trade['t2_stop_price'], invalidation_stop_price) if direction == 'SHORT' 
                                                  else max(current_trade['t2_stop_price'], invalidation_stop_price))
            
            # Re-check stops with new invalidation prices for *this current bar's H/L*
            if current_trade['t1_status'] == 'OPEN':
                if (direction == 'SHORT' and bar_high >= current_trade['t1_stop_price']) or \
                   (direction == 'LONG' and bar_low <= current_trade['t1_stop_price']):
                    close_portion('t1', current_trade['t1_current_size'], current_trade['t1_stop_price'], "STOPPED_OUT_POST_INVALIDATION")

            if current_trade['t2_status'] != 'CLOSED' and current_trade['t2_current_size'] > 1e-9:
                if (direction == 'SHORT' and bar_high >= current_trade['t2_stop_price']) or \
                   (direction == 'LONG' and bar_low <= current_trade['t2_stop_price']):
                    close_portion('t2', current_trade['t2_current_size'], current_trade['t2_stop_price'], "STOPPED_OUT_POST_INVALIDATION")

            if math.isclose(current_trade.get('t1_current_size', 0.0), 0.0, abs_tol=1e-9) and \
               math.isclose(current_trade.get('t2_current_size', 0.0), 0.0, abs_tol=1e-9):
                current_trade['active'] = False
                return current_trade, trade_actions, pnl_for_this_bar

    # --- III. Check Scale-Out & Profit-Taking Exits (PRD 3.4) ---
    # Targets are based on decision_vwap & decision_sigma (known at start of bar)
    # Hits are checked against current bar's H/L

    # Target 1: Price reaches decision_vwap ± SCALE_OUT_1_SIGMA_TARGET * decision_sigma
    target_1sigma_price = (decision_vwap - (SCALE_OUT_1_SIGMA_TARGET * decision_sigma) if direction == 'SHORT' 
                           else decision_vwap + (SCALE_OUT_1_SIGMA_TARGET * decision_sigma))
    
    if not current_trade.get('scaled_out_t2_50_pct', False) and current_trade['t2_current_size'] > 1e-9:
        if (direction == 'SHORT' and bar_low <= target_1sigma_price) or \
           (direction == 'LONG' and bar_high >= target_1sigma_price):
            
            amount_to_close_t2 = current_trade['t2_entry_size'] * 0.50 # 50% of *original* T2
            close_portion('t2', amount_to_close_t2, target_1sigma_price, "SCALED_OUT_50PCT")
            current_trade['scaled_out_t2_50_pct'] = True
            current_trade['vwap_1sigma_target_hit_for_t1'] = True # Mark objective conceptually met for T1 path

            # Trail REMAINING T2 to decision_vwap +/- SCALE_OUT_1_T2_TRAIL_SIGMA * decision_sigma
            if current_trade['t2_current_size'] > 1e-9: # If still T2 left
                new_t2_trail_stop = (decision_vwap + (SCALE_OUT_1_T2_TRAIL_SIGMA * decision_sigma) if direction == 'SHORT' 
                                     else decision_vwap - (SCALE_OUT_1_T2_TRAIL_SIGMA * decision_sigma))
                # Update T2 stop (never widen from current protective position, but this is a specific trail)
                current_trade['t2_stop_price'] = (min(current_trade['t2_stop_price'], new_t2_trail_stop) if direction == 'SHORT'
                                                  else max(current_trade['t2_stop_price'], new_t2_trail_stop))
                action_msg = f"T2_STOP_TRAILED_POST_SCALE1_TO_{current_trade['t2_stop_price']:.2f}"
                trade_actions.append(action_msg)
                current_trade['log'].append(f"{bar_data.name}: {action_msg}")

    # Target 2: Price touches decision_vwap
    if (direction == 'SHORT' and bar_low <= decision_vwap) or \
       (direction == 'LONG' and bar_high >= decision_vwap):
        if current_trade['t1_status'] == 'OPEN' and current_trade['t1_current_size'] > 1e-9:
            close_portion('t1', current_trade['t1_current_size'], decision_vwap, "CLOSED_AT_VWAP")
            current_trade['vwap_1sigma_target_hit_for_t1'] = True # Ensure marked

        if not current_trade.get('scaled_out_t2_next_25_pct', False) and current_trade['t2_current_size'] > 1e-9:
            amount_to_close_t2_further = current_trade['t2_entry_size'] * 0.25 # 25% of *original*
            close_portion('t2', amount_to_close_t2_further, decision_vwap, "SCALED_OUT_FURTHER_25PCT_AT_VWAP")
            current_trade['scaled_out_t2_next_25_pct'] = True
            
    # Target 3: Price overshoots decision_vwap by –OVERSHOOT_SIGMA_TARGET * decision_sigma (short) / + (long)
    if current_trade['t2_current_size'] > 1e-9: # Only if there's T2 left for this specific trail
        overshoot_target_price = (decision_vwap - (OVERSHOOT_SIGMA_TARGET * decision_sigma) if direction == 'SHORT' 
                                  else decision_vwap + (OVERSHOOT_SIGMA_TARGET * decision_sigma))
        new_trail_stop_t2_overshoot = np.nan
        
        if (direction == 'SHORT' and bar_low <= overshoot_target_price):
            stop_anchor1_vwap = decision_vwap 
            stop_anchor2_price_plus_sigma = bar_close + (OVERSHOOT_TRAIL_SIGMA * decision_sigma) 
            new_trail_stop_t2_overshoot = max(stop_anchor1_vwap, stop_anchor2_price_plus_sigma) # For SHORT, stop is above price
            current_trade['t2_stop_price'] = min(current_trade['t2_stop_price'], new_trail_stop_t2_overshoot) # Tighten stop

        elif (direction == 'LONG' and bar_high >= overshoot_target_price):
            stop_anchor1_vwap = decision_vwap
            stop_anchor2_price_minus_sigma = bar_close - (OVERSHOOT_TRAIL_SIGMA * decision_sigma)
            new_trail_stop_t2_overshoot = min(stop_anchor1_vwap, stop_anchor2_price_minus_sigma) # For LONG, stop is below price
            current_trade['t2_stop_price'] = max(current_trade['t2_stop_price'], new_trail_stop_t2_overshoot) # Tighten stop
        
        if not pd.isna(new_trail_stop_t2_overshoot) and not math.isclose(new_trail_stop_t2_overshoot, current_trade['t2_stop_price'], rel_tol=1e-5):
            action_msg = f"T2_RESIDUAL_TRAILED_OVERSHOOT_TO_{current_trade['t2_stop_price']:.2f}"
            trade_actions.append(action_msg)
            current_trade['log'].append(f"{bar_data.name}: {action_msg}")

    # --- IV. Dynamic Trailing Stop Logic (PRD 3.3 - Anchor update, Never widen, Buffer floor) ---
    # This applies at the END of the bar processing for any open portions, using decision_vwap/sigma/atr
    # for anchors and current bar_close for buffer calculations.
    
    current_trade['bars_held'] = current_trade.get('bars_held', 0) + 1 
    # Comment: Timeout check uses >= TIMEOUT_BARS. Entry bar is bar 1 of holding.
    # So, if TIMEOUT_BARS=16, on the 16th bar *after* entry, bars_held becomes 16.
    # The trade is held for 15 full bars, and on the 16th bar's processing, it's timed out.
    # This is 16 * 30 min = 8 hours total, including the partial entry bar. This matches PRD.

    # For T1 (if still open):
    if current_trade['t1_status'] == 'OPEN':
        new_t1_anchor_stop = (entry_price + (T1_PROBE_STOP_SIGMA_OFFSET * decision_sigma) if direction == 'SHORT' 
                              else entry_price - (T1_PROBE_STOP_SIGMA_OFFSET * decision_sigma))
        
        potential_t1_stop = (min(current_trade['t1_stop_price'], new_t1_anchor_stop) if direction == 'SHORT' 
                             else max(current_trade['t1_stop_price'], new_t1_anchor_stop))
        
        min_stop_dist_t1 = ATR_BUFFER_MULTIPLIER * decision_atr # Use decision_atr (from prev bar)
        final_t1_stop = potential_t1_stop
        if direction == 'SHORT': # Stop is above price
            if (potential_t1_stop - bar_close) < min_stop_dist_t1 and decision_atr > 1e-9: # Check ATR is valid
                final_t1_stop = bar_close + min_stop_dist_t1
                final_t1_stop = min(final_t1_stop, current_trade['t1_stop_price']) # Ensure buffer doesn't widen past original stop
        else: # LONG, stop is below price
            if (bar_close - potential_t1_stop) < min_stop_dist_t1 and decision_atr > 1e-9:
                final_t1_stop = bar_close - min_stop_dist_t1
                final_t1_stop = max(final_t1_stop, current_trade['t1_stop_price'])

        if not math.isclose(final_t1_stop, current_trade['t1_stop_price'], rel_tol=1e-5):
             action_msg = f"T1_STOP_TRAILED_TO_{final_t1_stop:.2f}"
             trade_actions.append(action_msg)
             current_trade['log'].append(f"{bar_data.name}: {action_msg}")
             current_trade['t1_stop_price'] = final_t1_stop

    # For T2 (General trailing if not under specific scale-out trail logic)
    # This trails T2 based on its original rule (VWAP_t + 3sigma_t) IF it hasn't been scaled out yet,
    # as scale-out rules apply their own specific trailing stops.
    if current_trade['t2_current_size'] > 1e-9 and not current_trade.get('scaled_out_t2_50_pct', False): 
        new_t2_anchor_stop = (decision_vwap + (T2_CORE_STOP_SIGMA_OFFSET * decision_sigma) if direction == 'SHORT' 
                              else decision_vwap - (T2_CORE_STOP_SIGMA_OFFSET * decision_sigma))
        
        potential_t2_stop = (min(current_trade['t2_stop_price'], new_t2_anchor_stop) if direction == 'SHORT' 
                             else max(current_trade['t2_stop_price'], new_t2_anchor_stop))

        min_stop_dist_t2 = ATR_BUFFER_MULTIPLIER * decision_atr
        final_t2_stop = potential_t2_stop
        if direction == 'SHORT':
            if (potential_t2_stop - bar_close) < min_stop_dist_t2 and decision_atr > 1e-9:
                final_t2_stop = bar_close + min_stop_dist_t2
                final_t2_stop = min(final_t2_stop, current_trade['t2_stop_price'])
        else: 
            if (bar_close - potential_t2_stop) < min_stop_dist_t2 and decision_atr > 1e-9:
                final_t2_stop = bar_close - min_stop_dist_t2
                final_t2_stop = max(final_t2_stop, current_trade['t2_stop_price'])
        
        if not math.isclose(final_t2_stop, current_trade['t2_stop_price'], rel_tol=1e-5):
            action_msg = f"T2_MAIN_STOP_TRAILED_TO_{final_t2_stop:.2f}"
            trade_actions.append(action_msg)
            current_trade['log'].append(f"{bar_data.name}: {action_msg}")
            current_trade['t2_stop_price'] = final_t2_stop
            
    # --- V. Check Time-Out Exit (PRD 3.4) ---
    if current_trade['bars_held'] >= TIMEOUT_BARS:
        if current_trade['t1_current_size'] > 1e-9:
            close_portion('t1', current_trade['t1_current_size'], bar_close, "TIMEOUT_EXIT")
        if current_trade['t2_current_size'] > 1e-9:
            close_portion('t2', current_trade['t2_current_size'], bar_close, "TIMEOUT_EXIT")
        current_trade['active'] = False # This will ensure it exits the main active trade loop
    
    # Final check if trade is now fully closed from any of above actions this bar
    if math.isclose(current_trade.get('t1_current_size', 0.0), 0.0, abs_tol=1e-9) and \
       math.isclose(current_trade.get('t2_current_size', 0.0), 0.0, abs_tol=1e-9):
        current_trade['active'] = False
            
    return current_trade, trade_actions, pnl_for_this_bar

# (if __name__ == '__main__': block for testing as before)
if __name__ == '__main__':
    print("Testing trading logic functions (placeholders)...")
    # Add simple test calls here if desired

