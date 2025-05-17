import asyncio
import json
import websockets
import logging
import pandas as pd
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

async def fetch_historical_klines(exchange, symbol, timeframe, limit=100):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Fetched historical klines for {symbol} {timeframe}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch historical klines for {symbol} {timeframe}: {e}")
        return None

async def generate_mock_klines(symbol, timeframe):
    logger.warning(f"Generating mock kline data for {symbol} {timeframe}")
    base_price = 50000.0 if 'BTCUSDT' in symbol else 150.0
    df = pd.DataFrame({
        'timestamp': [pd.Timestamp.now()],
        'open': [base_price],
        'high': [base_price * 1.002],
        'low': [base_price * 0.998],
        'close': [base_price * 1.001],
        'volume': [100.0]
    })
    return df

async def start_streams(symbols, timeframes, strategy):
    uri = "wss://stream.binancefuture.com/stream"
    streams = [
        f"{symbol.lower()}@kline_{tf}" for symbol in symbols for tf in timeframes
    ] + [
        f"{symbol.lower()}@ticker" for symbol in symbols
    ]
    stream_param = "/".join(streams)
    full_url = f"{uri}?streams={stream_param}"

    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.set_sandbox_mode(True)

    reconnect_delay = 5
    max_reconnect_delay = 120
    reconnect_attempts = 0
    max_attempts = 3

    while True:
        if reconnect_attempts >= max_attempts:
            logger.debug(f"Max WebSocket attempts ({max_attempts}) reached, switching to REST API fallback")
            for symbol in symbols:
                for tf in timeframes:
                    df = await fetch_historical_klines(exchange, symbol, tf)
                    if df is not None:
                        await strategy.process_timeframe_data(symbol, tf, df)
                    else:
                        logger.debug(f"REST API failed for {symbol} {tf}, using mock data")
                        df = await generate_mock_klines(symbol, tf)
                        await strategy.process_timeframe_data(symbol, tf, df)
            await asyncio.sleep(60)
            reconnect_attempts = 0
            reconnect_delay = 5
            continue

        try:
            async with websockets.connect(full_url, ping_interval=None) as ws:
                logger.info(f"WebSocket connected to Binance Futures: {stream_param}")
                reconnect_attempts = 0
                reconnect_delay = 5
                while True:
                    response = await ws.recv()
                    message = json.loads(response)
                    data = message.get('data', {})
                    stream_name = message.get('stream', '')
                    if not data or not stream_name:
                        continue

                    symbol = data.get('s', '').upper()
                    if not symbol:
                        continue

                    if '@kline_' in stream_name:
                        kline = data.get('k', {})
                        if not kline:
                            continue
                        timeframe = stream_name.split('@kline_')[1]
                        df = pd.DataFrame({
                            'timestamp': [pd.Timestamp(kline['t'], unit='ms')],
                            'open': [float(kline['o'])],
                            'high': [float(kline['h'])],
                            'low': [float(kline['l'])],
                            'close': [float(kline['c'])],
                            'volume': [float(kline['v'])]
                        })
                        await strategy.process_timeframe_data(symbol, timeframe, df)

                    elif '@ticker' in stream_name:
                        price = float(data.get('c', 0))
                        if price > 0:
                            await strategy.process_tick(symbol, price)

        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"WebSocket closed: {e}. Reconnecting in {reconnect_delay} seconds...")
        except Exception as e:
            logger.error(f"WebSocket error: {e}. Reconnecting in {reconnect_delay} seconds...")
        reconnect_attempts += 1
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    await exchange.close()
