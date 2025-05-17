import asyncio
import json
import logging
from live_strategy import LiveMAStrategy
from binance_ws import start_streams
from rich.console import Console
from api_client import BinanceClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    console = Console()
    with open('config.json', 'r') as f:
        config = json.load(f)
    symbols = config.get('symbols', [])
    timeframes = config.get('timeframes', [])
    testnet = config.get('testnet', True)

    # Instantiate the Binance client
    client = BinanceClient(testnet=testnet)

    strategy = LiveMAStrategy(client, symbols, timeframes)
    await strategy.async_init()

    stream_task = asyncio.create_task(start_streams(symbols, timeframes, strategy))

    try:
        while True:
            strategy.display_status()
            await asyncio.sleep(300)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down gracefully")
        stream_task.cancel()
        await client.close()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        stream_task.cancel()
        await client.close()
    finally:
        await asyncio.sleep(0.1)

if __name__ == '__main__':
    asyncio.run(main())
