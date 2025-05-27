# File: src/core/trading_logic.py

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
    slip_fixed = (fixed_percentage / 100.0) * execution_price 
    slip_spread_component = 0.5 * (spread_percentage / 100.0) * execution_price
    return max(slip_fixed, slip_spread_component)

def check_entry_signal(bar_data): 
    signal = None
    entry_price_candidate = None 

    vwap_upper2_signal = bar_data.get('vwap_upper2_lagged', float('inf'))
    vwap_lower2_signal = bar_data.get('vwap_lower2_lagged', float('-inf'))
    z_score_signal = bar_data.get('volume_delta_zscore_lagged', 0.0)

    if bar_data['high'] >= vwap_upper2_signal and z_score_signal >= Z_SCORE_THRESHOLD_SHORT:
        signal = 'SHORT'
        entry_price_candidate = vwap_upper2_signal 
    elif bar_data['low'] <= vwap_lower2_signal and z_score_signal <= Z_SCORE_THRESHOLD_LONG:
        signal = 'LONG'
        entry_price_candidate = vwap_lower2_signal
            
    return signal, entry_price_candidate

def calculate_initial_stop_prices(entry_price_after_slippage, 
                                  vwap_at_entry_lagged,
                                  sigma_at_entry_lagged,
                                  trade_direction):
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
    risk_budget_dollars = available_equity_for_sizing * tranche_risk_fraction
    stop_distance_dollars = abs(entry_price_after_slippage - stop_price)

    if math.isclose(stop_distance_dollars, 0, abs_tol=1e-9):
        return 0.0
            
    raw_size_units = risk_budget_dollars / stop_distance_dollars

    vol_adj = 1.0
    if vol_adj_method == 'ATR':
        if current_atr_lagged > 1e-9 and median_atr_lagged > 1e-9 : 
             vol_adj = median_atr_lagged / current_atr_lagged
    
    vol_adj = min(max(vol_adj, VOL_ADJ_MIN_CAP), VOL_ADJ_MAX_CAP)
    position_size_units = raw_size_units * vol_adj
    return position_size_units

def update_trade_status_per_bar(current_trade, bar_data):
    if not current_trade.get('active', False): # Check 'active' key safely
        return current_trade, [], 0.0

    trade_actions = []
    pnl_for_this_bar = 0.0
    
    direction = current_trade['direction']
    # CORRECTED KEY: Use 'actual_entry_price' which is the fill price after slippage
    # This is the price P&L should be based on.
    entry_price_fill = current_trade['actual_entry_price'] 
    initial_sigma0 = current_trade['initial_sigma0'] 

    bar_high = bar_data['high']
    bar_low = bar_data['low']
    bar_close = bar_data['close']
    
    decision_vwap = bar_data.get('session_vwap_lagged', current_trade['initial_vwap0']) 
    decision_sigma = bar_data.get('session_stdev_lagged', current_trade['initial_sigma0'])
    decision_zscore = bar_data.get('volume_delta_zscore_lagged', 0.0)
    decision_atr = bar_data.get('atr_20_30m_lagged', 0.0001)

    def close_portion(tranche_label, amount_to_close, actual_exit_price, reason):
        nonlocal pnl_for_this_bar
        size_key_current = f"{tranche_label}_current_size"
        status_key = f"{tranche_label}_status"
        
        if current_trade.get(size_key_current, 0.0) <= 1e-9: return
        actual_amount_closed = min(amount_to_close, current_trade[size_key_current])
        if actual_amount_closed <= 1e-9: return

        slippage_per_unit = calculate_slippage_amount_per_unit(actual_exit_price)
        effective_exit_price = actual_exit_price + slippage_per_unit if direction == 'SHORT' else actual_exit_price - slippage_per_unit
        
        # P&L calculation uses entry_price_fill (the actual fill price of the trade)
        portion_pnl = (entry_price_fill - effective_exit_price) * actual_amount_closed if direction == 'SHORT' else (effective_exit_price - entry_price_fill) * actual_amount_closed
        
        pnl_for_this_bar += portion_pnl
        current_trade['pnl'] += portion_pnl
        current_trade[size_key_current] -= actual_amount_closed
        
        action_detail = (f"{tranche_label}_{reason} @ {actual_exit_price:.2f} "
                         f"(eff: {effective_exit_price:.2f}), "
                         f"Closed: {actual_amount_closed:.8f}, PnL: {portion_pnl:.2f}")
        trade_actions.append(action_detail)
        current_trade['log'].append(f"{bar_data.name}: {action_detail}")

        if math.isclose(current_trade[size_key_current], 0.0, abs_tol=1e-9):
            current_trade[size_key_current] = 0.0
            current_trade[status_key] = 'CLOSED'
            trade_actions.append(f"{tranche_label}_FULLY_CLOSED")
            current_trade['log'].append(f"{bar_data.name}: {tranche_label}_FULLY_CLOSED")

    # --- I. Check for Stop-Loss Hits ---
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

    if math.isclose(current_trade.get('t1_current_size', 0.0), 0.0, abs_tol=1e-9) and \
       math.isclose(current_trade.get('t2_current_size', 0.0), 0.0, abs_tol=1e-9):
        current_trade['active'] = False
        return current_trade, trade_actions, pnl_for_this_bar

    # --- II. Check Invalidation Rule (PRD 3.3) ---
    if not current_trade.get('vwap_1sigma_target_hit_for_t1', False) and \
       (current_trade['t1_status'] == 'OPEN' or current_trade['t2_current_size'] > 1e-9):
        z_flipped = (direction == 'SHORT' and decision_zscore < 0) or \
                    (direction == 'LONG' and decision_zscore > 0)
        if z_flipped:
            # Invalidation stop is based on the original fill price (entry_price_fill)
            invalidation_stop_price = (entry_price_fill + (INVALIDATION_STOP_SIGMA_OFFSET * initial_sigma0) if direction == 'SHORT' 
                                       else entry_price_fill - (INVALIDATION_STOP_SIGMA_OFFSET * initial_sigma0))
            action_msg = f"INVALIDATION_Z_FLIP_STOPS_MOVED_TO_{invalidation_stop_price:.2f}"
            trade_actions.append(action_msg)
            current_trade['log'].append(f"{bar_data.name}: {action_msg}")

            if current_trade['t1_status'] == 'OPEN':
                current_trade['t1_stop_price'] = (min(current_trade['t1_stop_price'], invalidation_stop_price) if direction == 'SHORT' 
                                                  else max(current_trade['t1_stop_price'], invalidation_stop_price))
            if current_trade['t2_current_size'] > 1e-9 and current_trade['t2_status'] != 'CLOSED':
                current_trade['t2_stop_price'] = (min(current_trade['t2_stop_price'], invalidation_stop_price) if direction == 'SHORT' 
                                                  else max(current_trade['t2_stop_price'], invalidation_stop_price))
            
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
    target_1sigma_price = (decision_vwap - (SCALE_OUT_1_SIGMA_TARGET * decision_sigma) if direction == 'SHORT' 
                           else decision_vwap + (SCALE_OUT_1_SIGMA_TARGET * decision_sigma))
    
    if not current_trade.get('scaled_out_t2_50_pct', False) and current_trade['t2_current_size'] > 1e-9:
        if (direction == 'SHORT' and bar_low <= target_1sigma_price) or \
           (direction == 'LONG' and bar_high >= target_1sigma_price):
            amount_to_close_t2 = current_trade['t2_entry_size'] * 0.50
            close_portion('t2', amount_to_close_t2, target_1sigma_price, "SCALED_OUT_50PCT")
            current_trade['scaled_out_t2_50_pct'] = True
            current_trade['vwap_1sigma_target_hit_for_t1'] = True

            if current_trade['t2_current_size'] > 1e-9:
                new_t2_trail_stop = (decision_vwap + (SCALE_OUT_1_T2_TRAIL_SIGMA * decision_sigma) if direction == 'SHORT' 
                                     else decision_vwap - (SCALE_OUT_1_T2_TRAIL_SIGMA * decision_sigma))
                current_trade['t2_stop_price'] = (min(current_trade['t2_stop_price'], new_t2_trail_stop) if direction == 'SHORT'
                                                  else max(current_trade['t2_stop_price'], new_t2_trail_stop))
                action_msg = f"T2_STOP_TRAILED_POST_SCALE1_TO_{current_trade['t2_stop_price']:.2f}"
                trade_actions.append(action_msg)
                current_trade['log'].append(f"{bar_data.name}: {action_msg}")

    if (direction == 'SHORT' and bar_low <= decision_vwap) or \
       (direction == 'LONG' and bar_high >= decision_vwap):
        if current_trade['t1_status'] == 'OPEN' and current_trade['t1_current_size'] > 1e-9:
            close_portion('t1', current_trade['t1_current_size'], decision_vwap, "CLOSED_AT_VWAP")
            current_trade['vwap_1sigma_target_hit_for_t1'] = True 

        if not current_trade.get('scaled_out_t2_next_25_pct', False) and current_trade['t2_current_size'] > 1e-9:
            amount_to_close_t2_further = current_trade['t2_entry_size'] * 0.25
            close_portion('t2', amount_to_close_t2_further, decision_vwap, "SCALED_OUT_FURTHER_25PCT_AT_VWAP")
            current_trade['scaled_out_t2_next_25_pct'] = True
            
    if current_trade['t2_current_size'] > 1e-9:
        overshoot_target_price = (decision_vwap - (OVERSHOOT_SIGMA_TARGET * decision_sigma) if direction == 'SHORT' 
                                  else decision_vwap + (OVERSHOOT_SIGMA_TARGET * decision_sigma))
        new_trail_stop_t2_overshoot = np.nan
        
        if (direction == 'SHORT' and bar_low <= overshoot_target_price):
            stop_anchor1_vwap = decision_vwap 
            stop_anchor2_price_plus_sigma = bar_close + (OVERSHOOT_TRAIL_SIGMA * decision_sigma) 
            new_trail_stop_t2_overshoot = max(stop_anchor1_vwap, stop_anchor2_price_plus_sigma)
            current_trade['t2_stop_price'] = min(current_trade['t2_stop_price'], new_trail_stop_t2_overshoot)

        elif (direction == 'LONG' and bar_high >= overshoot_target_price):
            stop_anchor1_vwap = decision_vwap
            stop_anchor2_price_minus_sigma = bar_close - (OVERSHOOT_TRAIL_SIGMA * decision_sigma)
            new_trail_stop_t2_overshoot = min(stop_anchor1_vwap, stop_anchor2_price_minus_sigma)
            current_trade['t2_stop_price'] = max(current_trade['t2_stop_price'], new_trail_stop_t2_overshoot)
        
        if not pd.isna(new_trail_stop_t2_overshoot) and not math.isclose(new_trail_stop_t2_overshoot, current_trade['t2_stop_price'], rel_tol=1e-5): # Check if actually updated
            action_msg = f"T2_RESIDUAL_TRAILED_OVERSHOOT_TO_{current_trade['t2_stop_price']:.2f}"
            trade_actions.append(action_msg)
            current_trade['log'].append(f"{bar_data.name}: {action_msg}")

    # --- IV. Dynamic Trailing Stop Logic ---
    current_trade['bars_held'] = current_trade.get('bars_held', 0) + 1 
    
    if current_trade['t1_status'] == 'OPEN':
        # T1 stop is anchored to entry_price_fill + offset * decision_sigma
        new_t1_anchor_stop = (entry_price_fill + (T1_PROBE_STOP_SIGMA_OFFSET * decision_sigma) if direction == 'SHORT' 
                              else entry_price_fill - (T1_PROBE_STOP_SIGMA_OFFSET * decision_sigma))
        potential_t1_stop = (min(current_trade['t1_stop_price'], new_t1_anchor_stop) if direction == 'SHORT' 
                             else max(current_trade['t1_stop_price'], new_t1_anchor_stop))
        
        min_stop_dist_t1 = ATR_BUFFER_MULTIPLIER * decision_atr
        final_t1_stop = potential_t1_stop
        if decision_atr > 1e-9: # Only apply buffer if ATR is valid
            if direction == 'SHORT':
                if (potential_t1_stop - bar_close) < min_stop_dist_t1 :
                    final_t1_stop = bar_close + min_stop_dist_t1
                    final_t1_stop = min(final_t1_stop, current_trade['t1_stop_price']) 
            else: 
                if (bar_close - potential_t1_stop) < min_stop_dist_t1:
                    final_t1_stop = bar_close - min_stop_dist_t1
                    final_t1_stop = max(final_t1_stop, current_trade['t1_stop_price'])

        if not math.isclose(final_t1_stop, current_trade['t1_stop_price'], rel_tol=1e-5):
             action_msg = f"T1_STOP_TRAILED_TO_{final_t1_stop:.2f}"
             trade_actions.append(action_msg)
             current_trade['log'].append(f"{bar_data.name}: {action_msg}")
             current_trade['t1_stop_price'] = final_t1_stop

    if current_trade['t2_current_size'] > 1e-9 and not current_trade.get('scaled_out_t2_50_pct', False): 
        new_t2_anchor_stop = (decision_vwap + (T2_CORE_STOP_SIGMA_OFFSET * decision_sigma) if direction == 'SHORT' 
                              else decision_vwap - (T2_CORE_STOP_SIGMA_OFFSET * decision_sigma))
        potential_t2_stop = (min(current_trade['t2_stop_price'], new_t2_anchor_stop) if direction == 'SHORT' 
                             else max(current_trade['t2_stop_price'], new_t2_anchor_stop))

        min_stop_dist_t2 = ATR_BUFFER_MULTIPLIER * decision_atr
        final_t2_stop = potential_t2_stop
        if decision_atr > 1e-9: # Only apply buffer if ATR is valid
            if direction == 'SHORT':
                if (potential_t2_stop - bar_close) < min_stop_dist_t2 :
                    final_t2_stop = bar_close + min_stop_dist_t2
                    final_t2_stop = min(final_t2_stop, current_trade['t2_stop_price'])
            else: 
                if (bar_close - potential_t2_stop) < min_stop_dist_t2:
                    final_t2_stop = bar_close - min_stop_dist_t2
                    final_t2_stop = max(final_t2_stop, current_trade['t2_stop_price'])
        
        if not math.isclose(final_t2_stop, current_trade['t2_stop_price'], rel_tol=1e-5):
            action_msg = f"T2_MAIN_STOP_TRAILED_TO_{final_t2_stop:.2f}"
            trade_actions.append(action_msg)
            current_trade['log'].append(f"{bar_data.name}: {action_msg}")
            current_trade['t2_stop_price'] = final_t2_stop
            
    # --- V. Check Time-Out Exit ---
    if current_trade['bars_held'] >= TIMEOUT_BARS:
        if current_trade['t1_current_size'] > 1e-9:
            close_portion('t1', current_trade['t1_current_size'], bar_close, "TIMEOUT_EXIT")
        if current_trade['t2_current_size'] > 1e-9:
            close_portion('t2', current_trade['t2_current_size'], bar_close, "TIMEOUT_EXIT")
        current_trade['active'] = False 
    
    if math.isclose(current_trade.get('t1_current_size', 0.0), 0.0, abs_tol=1e-9) and \
       math.isclose(current_trade.get('t2_current_size', 0.0), 0.0, abs_tol=1e-9):
        current_trade['active'] = False
            
    return current_trade, trade_actions, pnl_for_this_bar

# (if __name__ == '__main__': block for testing as before)
if __name__ == '__main__':
    print("Testing trading logic functions (placeholders)...")
    # Add simple test calls here if desired

