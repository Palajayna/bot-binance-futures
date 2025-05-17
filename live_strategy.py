import pandas as pd
import numpy as np
import talib
import logging
import asyncio
import datetime
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

class LiveMAStrategy:
    def __init__(self, client, symbols, timeframes):
        self.client = client
        self.symbols = symbols
        self.timeframes = timeframes
        self.leverage = 20
        # Ajuste: TP/SL como % do capital/margem, NÃO do preço!
        self.tp_roi = 5.0    # 5% de lucro sobre a margem/capital
        self.sl_roi = 3.5    # 3.5% de perda sobre a margem/capital
        self.maintenance_margin_rate = 0.005
        self.console = Console()
        self.in_position = {symbol: False for symbol in symbols}
        self.position_side = {symbol: None for symbol in symbols}
        self.entry_price = {symbol: 0.0 for symbol in symbols}
        self.sl_price = {symbol: 0.0 for symbol in symbols}
        self.tp_price = {symbol: 0.0 for symbol in symbols}
        self.quantity = {symbol: 0.0 for symbol in symbols}
        self.latest_close = {symbol: 0.0 for symbol in symbols}
        self.data = {symbol: {tf: pd.DataFrame() for tf in timeframes} for symbol in symbols}
        self.last_signal = {symbol: {tf: None for tf in timeframes} for symbol in symbols}
        self.unrealized_pnl = {symbol: 0.0 for symbol in symbols}
        self.realized_pnl = {symbol: 0.0 for symbol in symbols}
        self.funding_fee = {symbol: 0.0 for symbol in symbols}
        self.commission = {symbol: 0.0 for symbol in symbols}
        self.margin_used = {symbol: 0.0 for symbol in symbols}
        self.liquidation_price = {symbol: 0.0 for symbol in symbols}
        self.margin_ratio = {symbol: 0.0 for symbol in symbols}
        self.position_timeframe = {symbol: None for symbol in symbols}
        self.update_positions_task = None
        # Adaptive logic
        self.performance_window = 30
        self.min_win_rate = 0.45
        self.disable_timeframes = {symbol: set() for symbol in symbols}
        self.last_adaptation = datetime.datetime.utcnow()
        self.debug_signals = True  # extra debugging
        self.force_entry = False   # for testing - set True to always enter

    async def async_init(self):
        for symbol in self.symbols:
            try:
                symbol_clean = symbol.replace('/', '')
                response = await self.client.exchange.fapiPrivatePostLeverage({
                    'symbol': symbol_clean,
                    'leverage': self.leverage
                })
                logger.info(f"Set leverage {self.leverage}x for {symbol}: {response}")
            except Exception as e:
                logger.error(f"Failed to set leverage for {symbol}: {e}")
        await self.fetch_initial_history()
        self.update_positions_task = asyncio.create_task(self.update_positions_loop())

    async def fetch_initial_history(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                limit = {
                    '15m': 672,
                    '30m': 336,
                    '1h': 168,
                    '4h': 42
                }.get(tf, 100)
                try:
                    ohlcv = await self.client.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open','high','low','close','volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.data[symbol][tf] = df
                    logger.info(f"Fetched {len(df)} candles for {symbol} {tf}")
                except Exception as e:
                    logger.warning(f"Failed to fetch history for {symbol} {tf}: {e}")

    async def update_positions_loop(self):
        while True:
            try:
                await self.fetch_positions()
            except Exception as e:
                logger.error(f"Failed to update positions: {e}")
            await asyncio.sleep(3)

    async def fetch_positions(self):
        try:
            positions = await self.client.exchange.fetch_positions()
            for pos in positions:
                symbol = pos['symbol']
                if symbol not in self.symbols:
                    continue
                qty = float(pos.get('contracts', 0) or pos.get('positionAmt', 0) or 0)
                entry = float(pos.get('entryPrice', 0))
                mark = float(pos.get('markPrice', pos.get('lastPrice', entry)))
                un_pnl = float(pos.get('unrealizedPnl', pos.get('unRealizedProfit', 0)))
                leverage = float(pos.get('leverage', self.leverage))
                notional = abs(qty * mark)
                margin = notional / leverage if leverage else 0.0
                maint = self.maintenance_margin_rate
                liq_price = 0.0
                if qty > 0:
                    denominator = leverage + 1 - maint * leverage
                    if denominator > 0:
                        liq_price = entry * leverage / denominator
                elif qty < 0:
                    denominator = leverage - 1 + maint * leverage
                    if denominator > 0:
                        liq_price = entry * leverage / denominator
                else:
                    liq_price = 0.0
                margin_ratio = abs(un_pnl) / margin * 100 if margin else 0.0
                self.in_position[symbol] = abs(qty) > 0
                self.position_side[symbol] = "long" if qty > 0 else ("short" if qty < 0 else None)
                self.entry_price[symbol] = entry if abs(qty) > 0 else 0.0
                self.quantity[symbol] = abs(qty)
                self.latest_close[symbol] = mark
                self.unrealized_pnl[symbol] = un_pnl
                self.margin_used[symbol] = margin
                self.liquidation_price[symbol] = liq_price
                self.margin_ratio[symbol] = margin_ratio
                funding = await self.fetch_funding_fee(symbol)
                self.funding_fee[symbol] = funding
                commission = await self.fetch_commission(symbol)
                self.commission[symbol] = commission
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")

    async def fetch_funding_fee(self, symbol):
        try:
            rows = await self.client.exchange.fapiPrivateGetIncome(
                {"symbol": symbol, "incomeType": "FUNDING_FEE", "limit": 10}
            )
            funding = sum(float(x['income']) for x in rows)
            return funding
        except Exception as e:
            logger.warning(f"Failed to fetch funding for {symbol}: {e}")
            return 0.0

    async def fetch_commission(self, symbol):
        try:
            trades = await self.client.exchange.fapiPrivateGetUserTrades(
                {"symbol": symbol, "limit": 10}
            )
            commission = sum(float(t['commission']) for t in trades)
            return commission
        except Exception as e:
            logger.warning(f"Failed to fetch commission for {symbol}: {e}")
            return 0.0

    async def process_timeframe_data(self, symbol, timeframe, df):
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if df is None or df.empty:
            logger.warning(f"Empty OHLCV data for {symbol} {timeframe}")
            return
        missing_cols = [col for col in required_columns if col not in df]
        if missing_cols:
            logger.warning(f"Missing columns {missing_cols} in OHLCV data for {symbol} {timeframe}")
            return
        try:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            self.data[symbol][timeframe] = pd.concat([self.data[symbol][timeframe], df]).drop_duplicates(subset=['timestamp']).tail(100)
            self.latest_close[symbol] = df['close'].iloc[-1]
            await self.evaluate_strategy(symbol)
        except Exception as e:
            logger.warning(f"Invalid OHLCV data format for {symbol} {timeframe}: {e}")

    async def process_tick(self, symbol, price):
        self.latest_close[symbol] = price

    def get_signal_for_timeframe(self, symbol, timeframe):
        df = self.data[symbol][timeframe]
        if len(df) < 35:
            if self.debug_signals:
                logger.info(f"Not enough data for {symbol} {timeframe} to check signal")
            return None
        close = df['close']
        rsi = talib.RSI(close, timeperiod=14)
        macd, macdsignal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        sma_fast = talib.SMA(close, timeperiod=9)
        sma_slow = talib.SMA(close, timeperiod=21)
        signal = None

        # LOOSER: Wider RSI, looser MACD, and allow price near SMA21
        if (
            rsi.iloc[-1] > 54 and
            macd.iloc[-1] > macdsignal.iloc[-1] - 10 and
            sma_fast.iloc[-1] > sma_slow.iloc[-1] * 0.99 and
            close.iloc[-1] > sma_slow.iloc[-1] * 0.995
        ):
            signal = "long"
        elif (
            rsi.iloc[-1] < 46 and
            macd.iloc[-1] < macdsignal.iloc[-1] + 10 and
            sma_fast.iloc[-1] < sma_slow.iloc[-1] * 1.01 and
            close.iloc[-1] < sma_slow.iloc[-1] * 1.005
        ):
            signal = "short"
        if self.force_entry:
            signal = "long"
        if self.debug_signals:
            logger.info(
                f"Signal for {symbol} {timeframe}: {signal} (RSI={rsi.iloc[-1]:.2f}, MACD={macd.iloc[-1]:.2f}, Signal={macdsignal.iloc[-1]:.2f}, SMA9={sma_fast.iloc[-1]:.2f}, SMA21={sma_slow.iloc[-1]:.2f}, Price={close.iloc[-1]:.2f})"
            )
        return signal

    async def evaluate_strategy(self, symbol):
        self.automate_learning(symbol)
        for tf in self.timeframes:
            if tf in self.disable_timeframes[symbol]:
                if self.debug_signals:
                    logger.info(f"[{symbol}][{tf}] Skipping disabled timeframe.")
                continue
            signal = self.get_signal_for_timeframe(symbol, tf)
            last_signal = self.last_signal[symbol][tf]
            if signal and not self.in_position[symbol]:
                qty = await self.calc_order_qty(symbol, self.latest_close[symbol])
                await self.open_position(symbol, signal, self.latest_close[symbol], qty)
                self.position_timeframe[symbol] = tf
                logger.info(f"{datetime.datetime.now()} Entry on {symbol} {tf} timeframe: {signal.upper()}")
            self.last_signal[symbol][tf] = signal
        if self.in_position[symbol]:
            tf = self.position_timeframe[symbol]
            signal = self.get_signal_for_timeframe(symbol, tf)
            side = self.position_side[symbol]
            price = self.latest_close[symbol]
            # Gatilho correto para TP/SL pelo preço com ROI desejado!
            if side == "long":
                if price <= self.sl_price[symbol] or price >= self.tp_price[symbol] or signal == "short":
                    await self.close_position(symbol, price)
            elif side == "short":
                if price >= self.sl_price[symbol] or price <= self.tp_price[symbol] or signal == "long":
                    await self.close_position(symbol, price)
            margin_ratio = self.margin_ratio.get(symbol, 0.0)
            if margin_ratio >= 80:
                logger.warning(f"{symbol} margin ratio {margin_ratio:.2f}%: CLOSE TO LIQUIDATION!")

    async def calc_order_qty(self, symbol, price):
        try:
            balance = await self.client.exchange.fetch_balance()
            usdt_bal = balance['total']['USDT']
            adaptive_risk = self.get_adaptive_risk(symbol)
            risk_amt = usdt_bal * adaptive_risk
            qty = (risk_amt * self.leverage) / price
            qty = round(qty, 3)
            logger.info(f"Calculated qty for {symbol}: {qty} (risk_amt={risk_amt}, price={price}, leverage={self.leverage})")
            return qty
        except Exception as e:
            logger.error(f"Failed to calc qty: {e}")
            return 0.0

    def get_adaptive_risk(self, symbol):
        win_rate = self.get_win_rate(symbol)
        min_risk = 0.005
        max_risk = 0.02
        base_risk = 0.01
        if win_rate is None:
            return base_risk
        if win_rate < 0.5:
            return max(min_risk, base_risk * 0.5)
        elif win_rate > 0.65:
            return min(max_risk, base_risk * 1.5)
        return base_risk

    async def open_position(self, symbol, side, price, qty):
        if qty <= 0:
            logger.warning(f"Qty zero, cannot open {side} for {symbol}")
            return
        try:
            order_side = 'buy' if side == "long" else 'sell'
            logger.info(f"Placing {side} order for {symbol}: qty={qty}, price={price}")
            order = await self.client.exchange.create_market_order(symbol, order_side, qty)
            logger.info(f"Opened {side} {symbol} {qty} @ {price}")
            self.in_position[symbol] = True
            self.position_side[symbol] = side
            self.entry_price[symbol] = price
            # CRUCIAL: TP/SL com base no ROI desejado (sobre a margem)
            lev = self.leverage
            tp_roi = self.tp_roi
            sl_roi = self.sl_roi
            if side == "long":
                self.tp_price[symbol] = price * (1 + tp_roi/100 / lev)
                self.sl_price[symbol] = price * (1 - sl_roi/100 / lev)
            else:
                self.tp_price[symbol] = price * (1 - tp_roi/100 / lev)
                self.sl_price[symbol] = price * (1 + sl_roi/100 / lev)
            self.quantity[symbol] = qty
            self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, price, "OPEN", 0.0)
        except Exception as e:
            logger.error(f"Failed to open position: {e}")

    async def close_position(self, symbol, price):
        try:
            side = self.position_side[symbol]
            qty = self.quantity[symbol]
            if qty <= 0:
                return
            order_side = 'sell' if side == "long" else 'buy'
            logger.info(f"Placing close {side} order for {symbol}: qty={qty}, price={price}")
            order = await self.client.exchange.create_market_order(symbol, order_side, qty)
            logger.info(f"Closed {side} {symbol} {qty} @ {price}")
            self.in_position[symbol] = False
            self.position_side[symbol] = None
            self.entry_price[symbol] = 0.0
            self.sl_price[symbol] = 0.0
            self.tp_price[symbol] = 0.0
            self.quantity[symbol] = 0.0
            self.realized_pnl[symbol] += self.unrealized_pnl.get(symbol, 0.0)
            self.log_trade(symbol, self.position_timeframe[symbol] or "unknown", side, self.entry_price[symbol], price, self.unrealized_pnl.get(symbol, 0.0))
            self.position_timeframe[symbol] = None
        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def log_trade(self, symbol, timeframe, side, entry_price, exit_price, pnl):
        result = "win" if isinstance(pnl, (float, int)) and pnl > 0 else "loss"
        try:
            with open("trade_log.csv", "a") as f:
                f.write(f"{datetime.datetime.now()},{symbol},{timeframe},{side},{entry_price},{exit_price},{pnl},{result}\n")
        except Exception as e:
            logger.warning(f"Failed to log trade: {e}")

    def get_recent_trades(self, symbol):
        try:
            df = pd.read_csv("trade_log.csv", header=None, names=[
                "datetime","symbol","timeframe","side","entry","exit","pnl","result"
            ])
            recent = df[df["symbol"] == symbol].tail(self.performance_window)
            return recent
        except Exception:
            return pd.DataFrame()

    def get_win_rate(self, symbol):
        trades = self.get_recent_trades(symbol)
        if trades.empty:
            return None
        return (trades['result'] == "win").mean()

    def automate_learning(self, symbol):
        now = datetime.datetime.utcnow()
        if (now - self.last_adaptation).total_seconds() < 600:
            return
        trades = self.get_recent_trades(symbol)
        if trades.empty:
            return
        grouped = trades.groupby("timeframe")["result"].value_counts().unstack(fill_value=0)
        for tf in self.timeframes:
            wins = grouped.loc[tf]["win"] if tf in grouped.index and "win" in grouped.columns else 0
            total = grouped.loc[tf].sum() if tf in grouped.index else 0
            win_rate = wins / total if total > 0 else 0
            if total >= 8 and win_rate < self.min_win_rate:
                self.disable_timeframes[symbol].add(tf)
                logger.info(f"Disabled {tf} for {symbol} (win rate {win_rate:.2f})")
            elif tf in self.disable_timeframes[symbol] and win_rate >= self.min_win_rate:
                self.disable_timeframes[symbol].remove(tf)
                logger.info(f"Re-enabled {tf} for {symbol} (win rate improved to {win_rate:.2f})")
        self.last_adaptation = now

    def display_status(self):
        table = Table(title=f"Strategy Status ({self.leverage}x Leverage)", show_lines=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Position", style="green")
        table.add_column("Side", style="magenta")
        table.add_column("Entry Price", style="yellow")
        table.add_column("Last Price", style="white")
        table.add_column("Stop Loss", style="red")
        table.add_column("Take Profit", style="blue")
        table.add_column("Qty", style="cyan")
        table.add_column("Unreal. PnL", style="green")
        table.add_column("PnL %", style="green")
        table.add_column("Realiz. PnL", style="green")
        table.add_column("Liq. Price", style="red")
        table.add_column("Margin", style="yellow")
        table.add_column("Mgn %", style="yellow")
        table.add_column("Funding", style="yellow")
        table.add_column("Comm.", style="red")
        table.add_column("Tfs OFF", style="bold red")
        table.add_column("TP PnL", style="green")
        table.add_column("SL PnL", style="red")
        for symbol in self.symbols:
            pos = "IN POSITION" if self.in_position.get(symbol) else "NO POSITION"
            side = self.position_side.get(symbol) or "-"
            entry = self.entry_price.get(symbol, 0.0)
            price = self.latest_close.get(symbol, 0.0)
            sl = self.sl_price.get(symbol, 0.0)
            tp = self.tp_price.get(symbol, 0.0)
            qty = self.quantity.get(symbol, 0.0)
            unpnl = self.unrealized_pnl.get(symbol, 0.0)
            margin = self.margin_used.get(symbol, 0.0)
            pnl_pct = f"{(unpnl/margin*100):.2f}%" if margin else "0.00%"
            rpnl = f"{self.realized_pnl.get(symbol, 0.0):.4f}"
            liq = f"{self.liquidation_price.get(symbol, 0.0):.2f}"
            mgn = f"{margin:.2f}"
            mgn_pct = f"{self.margin_ratio.get(symbol, 0.0):.2f}%"
            fund = f"{self.funding_fee.get(symbol, 0.0):.4f}"
            comm = f"{self.commission.get(symbol, 0.0):.4f}"
            tfs_off = ",".join(self.disable_timeframes[symbol]) if self.disable_timeframes[symbol] else "-"

            # Calcular PnL esperado para TP/SL (em USDT, igual ao Binance)
            if qty > 0:
                if side == "long":
                    pnl_tp = (tp - entry) * qty
                    pnl_sl = (sl - entry) * qty
                elif side == "short":
                    pnl_tp = (entry - tp) * qty
                    pnl_sl = (entry - sl) * qty
                else:
                    pnl_tp = pnl_sl = 0.0
            else:
                pnl_tp = pnl_sl = 0.0

            table.add_row(
                symbol, pos, side, f"{entry}", f"{price}", f"{sl}", f"{tp}", f"{qty}", f"{unpnl:.4f}",
                pnl_pct, rpnl, liq, mgn, mgn_pct, fund, comm, tfs_off,
                f"{pnl_tp:.2f}", f"{pnl_sl:.2f}"
            )
        self.console.print(table)

    def enable_force_entry(self):
        self.force_entry = True
        logger.warning("Force entry is ENABLED: bot will always try to enter a long position for testing.")

    def disable_force_entry(self):
        self.force_entry = False
        logger.warning("Force entry is DISABLED: bot will only enter on real signals.")

    def print_signals(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                signal = self.get_signal_for_timeframe(symbol, tf)
                print(f"{symbol} {tf}: {signal}")
