# File: src/backtester/engine.py

import pandas as pd
from src.core import indicators, trading_logic, trade 
import mplfinance as mpf
import math 
import numpy as np # For np.nan

MAX_ACCOUNT_RISK_PER_TRADE_FRACTION = 0.005
# Example: Max 5% of total equity can be at risk across all (currently 1) open trades.
# This is more relevant if max_concurrent_trades > 1.
# For a single trade system, the per-trade limit (0.5%) is the main constraint.
# MAX_TOTAL_COMMITTED_RISK_FRACTION = 0.05 

class BacktestEngine:
    def __init__(self, data_df, initial_equity=100000):
        self.data_df = data_df
        self.initial_equity = initial_equity
        self.account_equity = initial_equity
        
        self.current_trade_state = trade.create_new_trade_state_dict(
            entry_timestamp=None, 
            direction=None, 
            entry_price_at_band=None, 
            actual_entry_price=None,    
            vwap0_lagged=None, 
            sigma0_lagged=None
        )
        self.current_trade_state['active'] = False
        
        self.trade_history = []
        start_timestamp = data_df.index[0] if not data_df.empty else pd.Timestamp.now(tz='UTC')
        self.equity_curve = [{'timestamp': start_timestamp, 'equity': initial_equity}]

        self.committed_risk_dollars = 0.0

    def run_simulation(self):
        print("[Engine] Running backtest simulation (Plotting & Causality v2)...")
        if self.data_df.empty:
            print("[ERROR] Data for backtest is empty!") # Corrected simple print
            return

        for bar_index, current_bar_data in self.data_df.iterrows():
            pnl_from_bar_actions = 0.0

            if self.current_trade_state['active']:
                # Pass current_bar_data (which now has lagged indicators for decisions, and current HLOC for checking hits)
                self.current_trade_state, actions, pnl_from_bar_actions = \
                    trading_logic.update_trade_status_per_bar(self.current_trade_state, current_bar_data)
                
                self.log_actions(bar_index, actions)
                self.account_equity += pnl_from_bar_actions

                if not self.current_trade_state['active']: 
                    closed_trade_pnl = self.current_trade_state['pnl']
                    initial_risk_of_closed_trade = self.current_trade_state.get('initial_total_risk_dollars', 0.0)
                    self.committed_risk_dollars -= initial_risk_of_closed_trade
                    self.committed_risk_dollars = max(0.0, self.committed_risk_dollars) 
                    
                    print(f"[Engine] Trade closed on {bar_index}. Final Trade PnL: {closed_trade_pnl:.2f}. Freed risk: {initial_risk_of_closed_trade:.2f}. Committed Risk Now: {self.committed_risk_dollars:.2f}")
                    self.trade_history.append(self.current_trade_state.copy())
                    self.current_trade_state = trade.create_new_trade_state_dict(None, None, None, None, None, None) # Adjusted
                    self.current_trade_state['active'] = False
            
            if not self.current_trade_state['active']:
                signal, entry_price_candidate_at_band = trading_logic.check_entry_signal(current_bar_data) 
                
                if signal:
                    entry_ts = bar_index
                    vwap_at_entry_lagged = current_bar_data.get('session_vwap_lagged', np.nan)
                    sigma_at_entry_lagged = current_bar_data.get('session_stdev_lagged', np.nan)

                    if pd.isna(vwap_at_entry_lagged) or pd.isna(sigma_at_entry_lagged) or pd.isna(entry_price_candidate_at_band):
                        self.log_actions(bar_index, [f"Signal {signal} but missing lagged VWAP/Sigma or entry candidate. No trade."])
                    else:
                        entry_slippage_per_unit = trading_logic.calculate_slippage_amount_per_unit(entry_price_candidate_at_band)
                        actual_entry_price_after_slippage = (entry_price_candidate_at_band - entry_slippage_per_unit if signal == 'SHORT' 
                                                             else entry_price_candidate_at_band + entry_slippage_per_unit)

                        stop_t1_price, stop_t2_price = trading_logic.calculate_initial_stop_prices(
                            actual_entry_price_after_slippage, vwap_at_entry_lagged, sigma_at_entry_lagged, signal
                        )
                        
                        atr_val_lagged = current_bar_data.get('atr_20_30m_lagged', 0.0001)
                        median_atr_val_lagged = current_bar_data.get('median_atr_20_30m_lagged', atr_val_lagged)
                        available_equity_for_sizing = self.account_equity - self.committed_risk_dollars 

                        if available_equity_for_sizing <= 0:
                             self.log_actions(bar_index, [f"Signal {signal} but no available equity ({available_equity_for_sizing:.2f}) for sizing. No trade."])
                        else:
                            size_t1 = trading_logic.calculate_position_size_for_tranche(
                                available_equity_for_sizing, trading_logic.RISK_PER_TRADE_T1_FRACTION, 
                                actual_entry_price_after_slippage, stop_t1_price, atr_val_lagged, median_atr_val_lagged
                            )
                            size_t2 = trading_logic.calculate_position_size_for_tranche(
                                available_equity_for_sizing, trading_logic.RISK_PER_TRADE_T2_FRACTION, 
                                actual_entry_price_after_slippage, stop_t2_price, atr_val_lagged, median_atr_val_lagged
                            )
                            
                            initial_risk_t1_dollars = abs(actual_entry_price_after_slippage - stop_t1_price) * size_t1 if size_t1 > 1e-9 else 0.0
                            initial_risk_t2_dollars = abs(actual_entry_price_after_slippage - stop_t2_price) * size_t2 if size_t2 > 1e-9 else 0.0
                            potential_new_trade_total_risk_dollars = initial_risk_t1_dollars + initial_risk_t2_dollars
                            max_risk_for_this_specific_trade = self.account_equity * MAX_ACCOUNT_RISK_PER_TRADE_FRACTION
                            
                            if potential_new_trade_total_risk_dollars > 1e-9 and \
                               potential_new_trade_total_risk_dollars <= max_risk_for_this_specific_trade and \
                               (self.committed_risk_dollars + potential_new_trade_total_risk_dollars <= self.account_equity * 1.0): # Ensure total committed doesn't exceed equity
                                
                                self.current_trade_state = trade.create_new_trade_state_dict(
                                    entry_timestamp=entry_ts, 
                                    direction=signal, 
                                    entry_price_at_band=entry_price_candidate_at_band, # Store this!
                                    actual_entry_price=actual_entry_price_after_slippage, 
                                    vwap0_lagged=vwap_at_entry_lagged, 
                                    sigma0_lagged=sigma_at_entry_lagged
                                )
                                self.current_trade_state['entry_bar_index'] = bar_index 
                                self.current_trade_state['initial_total_risk_dollars'] = potential_new_trade_total_risk_dollars
                                
                                if size_t1 > 1e-9:
                                    self.current_trade_state['t1_entry_size'] = size_t1
                                    self.current_trade_state['t1_current_size'] = size_t1
                                    self.current_trade_state['t1_stop_price'] = stop_t1_price
                                    self.current_trade_state['t1_status'] = 'OPEN'
                                else: self.current_trade_state['t1_status'] = 'NOT_OPENED'

                                if size_t2 > 1e-9:
                                    self.current_trade_state['t2_entry_size'] = size_t2
                                    self.current_trade_state['t2_current_size'] = size_t2
                                    self.current_trade_state['t2_stop_price'] = stop_t2_price
                                    self.current_trade_state['t2_status'] = 'OPEN'
                                else: self.current_trade_state['t2_status'] = 'NOT_OPENED'

                                if self.current_trade_state['t1_status'] != 'OPEN' and self.current_trade_state['t2_status'] != 'OPEN':
                                    self.current_trade_state['active'] = False
                                else:
                                    self.current_trade_state['active'] = True
                                    self.committed_risk_dollars += potential_new_trade_total_risk_dollars
                                    entry_log_msg = (f"NEW_TRADE: {signal} @ Band:{entry_price_candidate_at_band:.2f} (Fill:{actual_entry_price_after_slippage:.2f}). "
                                                     f"Risk: {potential_new_trade_total_risk_dollars:.2f}. CommitRiskNow: {self.committed_risk_dollars:.2f}. "
                                                     f"T1 Sz: {size_t1:.8f}, SL: {stop_t1_price:.2f}. "
                                                     f"T2 Sz: {size_t2:.8f}, SL: {stop_t2_price:.2f}")
                                    self.log_actions(bar_index, [entry_log_msg])
                                    self.current_trade_state['log'].append(f"{bar_index}: {entry_log_msg}")
                            else:
                                self.log_actions(bar_index, [f"Signal {signal} but risk/size fail. PotRisk:{potential_new_trade_total_risk_dollars:.2f} MaxAllow:{max_risk_for_this_specific_trade:.2f}"])
            
            self.equity_curve.append({'timestamp': bar_index, 'equity': self.account_equity})
        
        print("[Engine] Simulation finished.")
        self.generate_report()
        self.plot_results()

    # ... (log_actions, generate_report, plot_results methods as previously defined)


    def log_actions(self, timestamp, actions):
        for action in actions:
            print(f"[{timestamp}] {action}")

    def generate_report(self):
        print("\n--- Backtest Performance Report ---")
        print(f"Initial Equity: {self.initial_equity:.2f}")
        print(f"Final Equity: {self.account_equity:.2f}")
        total_pnl = self.account_equity - self.initial_equity
        print(f"Total Net P&L: {total_pnl:.2f} ({ (total_pnl/self.initial_equity)*100 :.2f}%)")

        num_trades = len(self.trade_history)
        print(f"Total Trades Taken (closed): {num_trades}")

        if num_trades > 0:
            winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
            losing_trades = sum(1 for t in self.trade_history if t['pnl'] < 0)
            
            print(f"Winning Trades: {winning_trades}")
            print(f"Losing Trades: {losing_trades}")
            if (winning_trades + losing_trades) > 0 :
                 win_rate_non_scratch = winning_trades / (winning_trades + losing_trades) * 100
                 print(f"Win Rate (P&L>0 vs P&L<0): {win_rate_non_scratch:.2f}%")
            
            equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
            if not equity_df.empty:
                peak = equity_df['equity'].expanding(min_periods=1).max()
                drawdown = (equity_df['equity'] - peak) / peak
                max_drawdown = drawdown.min()
                print(f"Max Drawdown: {max_drawdown*100:.2f}%")
            else:
                print("Equity curve is empty, cannot calculate Max Drawdown.")


        print("-----------------------------------------------")

    def plot_results(self, num_bars_to_plot=200, plot_volume=True):
        print("[Engine] Generating results chart with trade markers (v2)...")
        # ... (initial checks for data_df, plot_df, missing_cols as before) ...
        if self.data_df.empty: # Simplified initial check
            print("[WARN] No data for plotting.")
            return
            
        plot_df = self.data_df.tail(num_bars_to_plot if num_bars_to_plot > 0 else len(self.data_df)).copy()
        if plot_df.empty:
            print("[WARN] Not enough data to plot.")
            return

        apds = []
        # VWAP and Bands (as before)
        apds.append(mpf.make_addplot(plot_df['session_vwap'], color='blue', width=0.7, panel=0, ylabel='VWAP'))
        apds.append(mpf.make_addplot(plot_df['vwap_upper2'], color='lime', width=0.7, panel=0, linestyle='dashed')) # Using actual vwap_upper2 for plotting bands
        apds.append(mpf.make_addplot(plot_df['vwap_lower2'], color='lime', width=0.7, panel=0, linestyle='dashed')) # Using actual vwap_lower2

        zscore_panel = 2 if plot_volume else 1
        apds.append(mpf.make_addplot(plot_df['volume_delta_zscore'], panel=zscore_panel, color='purple', ylabel='Vol Δ Z-Score'))
        apds.append(mpf.make_addplot(pd.Series(1.0, index=plot_df.index), panel=zscore_panel, color='gray', linestyle='dotted', width=0.7))
        apds.append(mpf.make_addplot(pd.Series(-1.0, index=plot_df.index), panel=zscore_panel, color='gray', linestyle='dotted', width=0.7))
        apds.append(mpf.make_addplot(pd.Series(0.0, index=plot_df.index), panel=zscore_panel, color='black', linestyle='solid', width=0.5))

        long_entries_marker_prices = pd.Series(np.nan, index=plot_df.index)
        long_exits_marker_prices = pd.Series(np.nan, index=plot_df.index)
        short_entries_marker_prices = pd.Series(np.nan, index=plot_df.index)
        short_exits_marker_prices = pd.Series(np.nan, index=plot_df.index)

        print(f"[Engine] Processing {len(self.trade_history)} closed trades for plotting markers...")
        for closed_trade in self.trade_history:
            entry_ts = closed_trade['entry_timestamp']
            # --- Use 'entry_price_at_band' for plotting the entry marker location ---
            plot_entry_price = closed_trade.get('entry_price_at_band', closed_trade['actual_entry_price']) # Fallback if somehow missing
            direction = closed_trade['direction']
            
            if entry_ts in plot_df.index:
                if direction == 'LONG':
                    # Plot arrow slightly below the low of the entry bar for visibility
                    long_entries_marker_prices.loc[entry_ts] = plot_df.loc[entry_ts, 'low'] * 0.998 
                elif direction == 'SHORT':
                    # Plot arrow slightly above the high of the entry bar for visibility
                    short_entries_marker_prices.loc[entry_ts] = plot_df.loc[entry_ts, 'high'] * 1.002
            
            for log_entry in closed_trade.get('log', []):
                if "NEW_TRADE" in log_entry: continue # Skip entry logs for exit markers

                log_timestamp_str_part = log_entry.split(":")[0]
                try:
                    log_ts_naive = pd.to_datetime(log_timestamp_str_part)
                    log_ts_aware = log_ts_naive.tz_localize(plot_df.index.tz if plot_df.index.tz else 'UTC') # Match plot_df timezone
                    
                    if log_ts_aware in plot_df.index:
                        exit_price_logged = np.nan
                        if "@ " in log_entry:
                            try:
                                price_str_part = log_entry.split("@ ")[1].split(" ")[0]
                                exit_price_logged = float(price_str_part) # This is the actual price level of the stop/target
                            except (IndexError, ValueError):
                                pass # Keep exit_price_logged as NaN
                        
                        if not pd.isna(exit_price_logged):
                            # Place exit marker at the actual exit price level
                            if direction == 'LONG': # Exit of a long
                                long_exits_marker_prices.loc[log_ts_aware] = exit_price_logged
                            elif direction == 'SHORT': # Exit of a short
                                short_exits_marker_prices.loc[log_ts_aware] = exit_price_logged
                except Exception as e:
                    print(f"[WARN] Plotter: Could not process log entry for exit marker: '{log_entry}'. Error: {e}")

        # Add marker plots
        if not long_entries_marker_prices.dropna().empty:
            apds.append(mpf.make_addplot(long_entries_marker_prices, type='scatter', color='green', marker='^', markersize=100, panel=0))
        if not long_exits_marker_prices.dropna().empty:
            apds.append(mpf.make_addplot(long_exits_marker_prices, type='scatter', color='blue', marker='v', markersize=70, panel=0))
        if not short_entries_marker_prices.dropna().empty:
            apds.append(mpf.make_addplot(short_entries_marker_prices, type='scatter', color='red', marker='v', markersize=100, panel=0))
        if not short_exits_marker_prices.dropna().empty:
            apds.append(mpf.make_addplot(short_exits_marker_prices, type='scatter', color='purple', marker='^', markersize=70, panel=0))

        num_panels = 3 if plot_volume else 2
        panel_ratios_val = (6, 1.5, 2.5) if plot_volume else (6, 2.5)

        try:
            mpf.plot(plot_df, type='candle', style='yahoo',
                     title=f'VolDelta-MR: {plot_df.index[0].date()} to {plot_df.index[-1].date()} ({len(plot_df)} bars)',
                     ylabel='Price', volume=plot_volume, addplot=apds,
                     panel_ratios=panel_ratios_val, figscale=1.5, figsize=(18,10),
                     # savefig=dict(fname='backtest_chart_trades.png', dpi=300) # Uncomment to save
                    )
        except Exception as e:
            print(f"[ERROR] Failed to generate mplfinance chart: {e}")
            import traceback
            traceback.print_exc()

# ... (if __name__ == '__main__': part of engine.py if you want to test it standalone)
