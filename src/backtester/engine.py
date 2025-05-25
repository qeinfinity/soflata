# File: src/backtester/engine.py

import pandas as pd
from src.core import indicators, trading_logic, trade # Ensure trade is imported
import mplfinance as mpf # Import mplfinance

class BacktestEngine:
    def __init__(self, data_df, initial_equity=100000):
        self.data_df = data_df # This IS our df_master_30min
        self.initial_equity = initial_equity
        self.account_equity = initial_equity
        
        self.current_trade_state = trade.create_new_trade_state_dict(None,None,None,None,None)
        self.current_trade_state['active'] = False
        
        self.trade_history = []
        # Initialize equity curve with the timestamp of the first bar of data_df
        # or a default if data_df is unexpectedly empty at initialization
        start_timestamp = data_df.index[0] if not data_df.empty else pd.Timestamp.now(tz='UTC')
        self.equity_curve = [{'timestamp': start_timestamp, 
                              'equity': initial_equity}]

        self.max_notional_allowed_based_on_24h_vol = float('inf')

        print(f"BacktestEngine initialized with initial equity: {initial_equity}")

    def run_simulation(self):
        print("[Engine] Running backtest simulation...")
        if self.data_df.empty:
            print("[ERROR] Data for backtest is empty in BacktestEngine!")
            return

        for bar_index, current_bar_data in self.data_df.iterrows():
            pnl_from_bar_actions = 0.0

            if self.current_trade_state['active']:
                self.current_trade_state, actions, pnl_from_bar_actions = trading_logic.update_trade_status_per_bar(
                    self.current_trade_state, current_bar_data 
                )
                self.log_actions(bar_index, actions)
                
                self.account_equity += pnl_from_bar_actions

                if not self.current_trade_state['active']:
                    print(f"[Engine] Trade closed on {bar_index}. Final Trade PnL: {self.current_trade_state['pnl']:.2f}")
                    self.trade_history.append(self.current_trade_state.copy())
                    self.current_trade_state = trade.create_new_trade_state_dict(None,None,None,None,None)
                    self.current_trade_state['active'] = False
            
            if not self.current_trade_state['active']:
                signal, entry_price_candidate = trading_logic.check_entry_signal(current_bar_data)
                if signal:
                    entry_ts = bar_index
                    vwap_at_entry = current_bar_data['session_vwap']
                    sigma_at_entry = current_bar_data['session_stdev']
                    
                    entry_slippage_per_unit = trading_logic.calculate_slippage_amount_per_unit(entry_price_candidate)
                    actual_entry_price = 0
                    if signal == 'SHORT':
                        actual_entry_price = entry_price_candidate - entry_slippage_per_unit
                    else: # LONG
                        actual_entry_price = entry_price_candidate + entry_slippage_per_unit

                    stop_t1_price, stop_t2_price = trading_logic.calculate_initial_stop_prices(
                        actual_entry_price, vwap_at_entry, sigma_at_entry, signal
                    )
                    
                    atr_val = current_bar_data.get('atr_20_30m', 0)
                    median_atr_val = current_bar_data.get('median_atr_20_30m', atr_val)

                    size_t1 = trading_logic.calculate_position_size_for_tranche(
                        self.account_equity, 0.002, actual_entry_price, stop_t1_price,
                        atr_val, median_atr_val, vol_adj_method='ATR'
                    )
                    size_t2 = trading_logic.calculate_position_size_for_tranche(
                        self.account_equity, 0.003, actual_entry_price, stop_t2_price,
                        atr_val, median_atr_val, vol_adj_method='ATR'
                    )
                    
                    if size_t1 > 0 or size_t2 > 0:
                        self.current_trade_state = trade.create_new_trade_state_dict(
                            entry_timestamp=entry_ts, 
                            direction=signal, 
                            entry_price=actual_entry_price, 
                            vwap0=vwap_at_entry, 
                            sigma0=sigma_at_entry
                        )
                        self.current_trade_state['entry_bar_index'] = bar_index
                        
                        if size_t1 > 0:
                            self.current_trade_state['t1_entry_size'] = size_t1
                            self.current_trade_state['t1_current_size'] = size_t1
                            self.current_trade_state['t1_stop_price'] = stop_t1_price
                            self.current_trade_state['t1_status'] = 'OPEN'
                        else:
                            self.current_trade_state['t1_status'] = 'NOT_OPENED'

                        if size_t2 > 0:
                            self.current_trade_state['t2_entry_size'] = size_t2
                            self.current_trade_state['t2_current_size'] = size_t2
                            self.current_trade_state['t2_stop_price'] = stop_t2_price
                            self.current_trade_state['t2_status'] = 'OPEN'
                        else:
                             self.current_trade_state['t2_status'] = 'NOT_OPENED'

                        if self.current_trade_state['t1_status'] != 'OPEN' and self.current_trade_state['t2_status'] != 'OPEN':
                            self.current_trade_state['active'] = False
                        else:
                            self.current_trade_state['active'] = True
                            entry_log_msg = (f"NEW_TRADE_ENTERED: {signal} @ {actual_entry_price:.2f}. "
                                             f"T1 Size: {size_t1:.4f}, SL: {stop_t1_price:.2f}. "
                                             f"T2 Size: {size_t2:.4f}, SL: {stop_t2_price:.2f}")
                            self.log_actions(bar_index, [entry_log_msg])
                            self.current_trade_state['log'].append(f"{bar_index}: {entry_log_msg}")
                    else:
                        self.log_actions(bar_index, [f"Signal {signal} but zero size for T1 & T2. No trade."])
            
            self.equity_curve.append({'timestamp': bar_index, 'equity': self.account_equity})
        
        print("[Engine] Simulation finished.")
        self.generate_report()
        self.plot_results() # Call the new plotting function

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
        """
        Plots the klines, VWAP with 2-sigma bands, and Volume Delta Z-Score.
        """
        print("[Engine] Generating results chart...")
        if self.data_df.empty:
            print("[WARN] No data available for plotting in BacktestEngine.")
            return
        if num_bars_to_plot <= 0:
            print("[WARN] num_bars_to_plot must be positive. Plotting all available data.")
            plot_df = self.data_df.copy()
        else:
            plot_df = self.data_df.tail(num_bars_to_plot).copy()

        if plot_df.empty:
            print("[WARN] Not enough data to plot the requested number of bars.")
            return

        # Ensure required columns exist
        required_plot_cols = ['open', 'high', 'low', 'close', 'session_vwap', 
                              'vwap_upper2', 'vwap_lower2', 'volume_delta_zscore']
        if plot_volume:
            required_plot_cols.append('volume')

        missing_cols = [col for col in required_plot_cols if col not in plot_df.columns]
        if missing_cols:
            print(f"[ERROR] Missing columns required for plotting: {missing_cols}. Cannot generate chart.")
            return

        # Prepare additional plots for mplfinance
        apds = [] # List to hold addplot dictionaries

        # VWAP and Bands
        apds.append(mpf.make_addplot(plot_df['session_vwap'], color='blue', width=0.7, panel=0, ylabel='VWAP'))
        apds.append(mpf.make_addplot(plot_df['vwap_upper2'], color='lime', width=0.7, panel=0, linestyle='dashed'))
        apds.append(mpf.make_addplot(plot_df['vwap_lower2'], color='lime', width=0.7, panel=0, linestyle='dashed'))

        # Volume Delta Z-Score in a new panel (e.g., panel 2, assuming volume is panel 1 if shown)
        # Panel 0 is main, Panel 1 is usually volume by default if `volume=True`
        zscore_panel = 2 if plot_volume else 1
        apds.append(mpf.make_addplot(plot_df['volume_delta_zscore'], panel=zscore_panel, color='purple', ylabel='Vol Δ Z-Score', secondary_y=False))
        
        # Add horizontal lines at +1 and -1 for Z-Score for reference (PRD Entry Condition)
        apds.append(mpf.make_addplot(pd.Series(1.0, index=plot_df.index), panel=zscore_panel, color='gray', linestyle='dotted', width=0.7))
        apds.append(mpf.make_addplot(pd.Series(-1.0, index=plot_df.index), panel=zscore_panel, color='gray', linestyle='dotted', width=0.7))
        apds.append(mpf.make_addplot(pd.Series(0.0, index=plot_df.index), panel=zscore_panel, color='black', linestyle='solid', width=0.5)) # Zero line


        # Optionally, plot trade entry/exit markers if self.trade_history is available and parsed
        # This is more advanced: requires iterating through self.trade_history and creating scatter plots
        # For now, we'll skip markers to keep it simpler.

        # Determine number of panels
        num_panels = 3 if plot_volume else 2

        # Plotting
        try:
            mpf.plot(plot_df, 
                     type='candle', 
                     style='yahoo', # or 'charles', 'binance', etc.
                     title=f'VolDelta-MR Backtest Results ({self.data_df.index.name if self.data_df.index.name else "Time"})', # Use index name if available
                     ylabel='Price',
                     volume=plot_volume, # Show volume in its own panel (panel 1)
                     addplot=apds,
                     panel_ratios=(6, 1, 3) if plot_volume else (6,3), # (main_chart, volume_panel, zscore_panel) or (main, zscore)
                     figscale=1.5, # Make plot larger
                     figsize=(16,9) # Specify figure size
                    )
            print("[Engine] Chart generated. If not showing, ensure you are in an interactive environment or save to file.")
        except Exception as e:
            print(f"[ERROR] Failed to generate mplfinance chart: {e}")
            import traceback
            traceback.print_exc()

# ... (if __name__ == '__main__': part of engine.py if you want to test it standalone)
