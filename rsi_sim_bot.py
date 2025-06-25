import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- CONFIGURATION ---
TRADING_PAIRS = [
    'ETHUSDT',
    'BTCUSDT',
    'SOLUSDT',
    'AVAXUSDT',
    'DOGEUSDT',
    'ADAUSDT',
    'LTCUSDT',
    'XRPUSDT'
]

KRAKEN_SYMBOLS = {
    'BTCUSDT': 'XBTUSDT',
    'ETHUSDT': 'ETHUSDT',
    'SOLUSDT': 'SOLUSDT',
    'AVAXUSDT': 'AVAXUSDT',
    'DOGEUSDT': 'DOGEUSDT',
    'ADAUSDT': 'ADAUSDT',
    'LTCUSDT': 'LTCUSDT',
    'XRPUSDT': 'XRPUSDT',
}

INTERVAL = 1  # in minutes
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
STARTING_BALANCE = 202.00
STOP_LOSS_PERCENT = 5.0
RECENT_DROP_THRESHOLD = 5.0
LOOKBACK_PERIOD = 15
MIN_TRADE_USD = 50
MAX_RETRIES = 3

portfolio = {
    'USDT': STARTING_BALANCE,
    'positions': {},
}

price_history = {pair: [] for pair in TRADING_PAIRS}

# --- RSI CALCULATION ---
def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Wilder’s smoothing method = EMA with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi[avg_loss == 0] = 100
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50

    return rsi

# --- OHLC FETCH FROM KRAKEN ---
def fetch_ohlc_kraken(pair, interval=1):
    kraken_pair = KRAKEN_SYMBOLS.get(pair)
    if not kraken_pair:
        print(f"[{datetime.now()}] No Kraken mapping for {pair}")
        return None

    url = 'https://api.kraken.com/0/public/OHLC'
    params = {'pair': kraken_pair, 'interval': interval}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if 'error' in data and data['error']:
            print(f"Error fetching {kraken_pair} from Kraken: {data['error']}")
            return None
        key = list(data['result'].keys())[0]
        ohlc = data['result'][key]
        df = pd.DataFrame(ohlc, columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching OHLC for {kraken_pair} from Kraken: {e}")
        return None

# --- OHLC FETCH FROM BINANCE (Fallback) ---
def fetch_ohlc_binance(pair, interval=1):
    binance_pair = pair.upper()
    interval_str = f"{interval}m"
    url = f'https://api.binance.com/api/v3/klines'
    params = {'symbol': binance_pair, 'interval': interval_str, 'limit': 100}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if isinstance(data, dict) and data.get('code'):
            print(f"Error fetching {pair} from Binance: {data.get('msg', 'Unknown error')}")
            return None
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        df['time'] = pd.to_datetime(df['close_time'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching OHLC for {pair} from Binance: {e}")
        return None

# --- FETCH DATA WITH RETRY AND FALLBACK ---
def fetch_data_with_retry(pair, interval=1):
    for attempt in range(1, MAX_RETRIES + 1):
        df = fetch_ohlc_kraken(pair, interval)
        if df is not None:
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            latest_rsi = df['rsi'].iloc[-1]
            if latest_rsi is not None and latest_rsi > 0:
                return df
            else:
                print(f"[{datetime.now()}] Attempt {attempt}: Invalid RSI for {pair}: {latest_rsi}")
        else:
            print(f"[{datetime.now()}] Attempt {attempt}: Failed to fetch data for {pair} from Kraken")

        time.sleep(1)

    print(f"[{datetime.now()}] Falling back to Binance for {pair}")
    df = fetch_ohlc_binance(pair, interval)
    if df is not None:
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        latest_rsi = df['rsi'].iloc[-1]
        if latest_rsi is not None and latest_rsi > 0:
            return df
        else:
            print(f"[{datetime.now()}] Invalid RSI from Binance for {pair}: {latest_rsi}")
    return None

# --- BUY LOGIC ---
def simulate_buy(pair, price, rsi, close_prices):
    global portfolio

    if rsi is None or rsi == 0 or np.isnan(rsi):
        print(f"[{datetime.now()}] Skipping buy {pair} due to invalid RSI: {rsi}")
        return False

    if rsi >= BUY_THRESHOLD:
        return False

    if pair in portfolio['positions']:
        return False

    history = price_history[pair]
    history.append(price)
    if len(history) > LOOKBACK_PERIOD:
        history.pop(0)

    recent_drop = ((price - history[0]) / history[0]) * 100 if len(history) == LOOKBACK_PERIOD else 0.0
    if recent_drop <= -RECENT_DROP_THRESHOLD:
        print(f"[{datetime.now()}] Skipping buy {pair} due to recent drop {recent_drop:.2f}%")
        return False

    volatility = np.std(close_prices[-RSI_PERIOD:]) if len(close_prices) >= RSI_PERIOD else 0
    allocation_pct = min(0.5, max(0.05, 0.2 / volatility)) if volatility > 0 else 0.1

    amount_to_spend = portfolio['USDT'] * allocation_pct
    if amount_to_spend < MIN_TRADE_USD:
        print(f"[{datetime.now()}] Skipping buy {pair} because allocation ${amount_to_spend:.2f} < minimum trade size ${MIN_TRADE_USD}")
        return False

    amount = amount_to_spend / price
    if amount > 0 and amount_to_spend <= portfolio['USDT']:
        portfolio['positions'][pair] = {'amount': amount, 'buy_price': price}
        portfolio['USDT'] -= amount_to_spend
        print(f"[{datetime.now()}] BUY {pair} @ ${price:.2f} | RSI={rsi:.2f} | Allocated: ${amount_to_spend:.2f}")
        return True

    print(f"[{datetime.now()}] Not enough USDT to buy {pair}")
    return False

# --- SELL LOGIC (includes stop loss) ---
def simulate_sell(pair, price, rsi):
    global portfolio

    if pair not in portfolio['positions']:
        return False

    position = portfolio['positions'][pair]
    loss_pct = ((price - position['buy_price']) / position['buy_price']) * 100

    if loss_pct <= -STOP_LOSS_PERCENT:
        portfolio['USDT'] += position['amount'] * price
        print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
        del portfolio['positions'][pair]
        return True

    if rsi > SELL_THRESHOLD:
        portfolio['USDT'] += position['amount'] * price
        profit = (price - position['buy_price']) * position['amount']
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${profit:.2f}")
        del portfolio['positions'][pair]
        return True

    return False

# --- MAIN LOOP ---
def main():
    print("Starting RSI simulation bot with volatility-based position sizing, retries, and minimum trade size...")
    max_buys_per_cycle = 1

    while True:
        buys_this_cycle = 0
        total_value = portfolio['USDT']

        for pair in TRADING_PAIRS:
            df = fetch_data_with_retry(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                print(f"[{datetime.now()}] Not enough data for {pair}, skipping...")
                continue

            latest_rsi = df['rsi'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            close_prices = df['close'].values

            recent_drop = ((latest_price - close_prices[-LOOKBACK_PERIOD]) / close_prices[-LOOKBACK_PERIOD]) * 100 if len(close_prices) >= LOOKBACK_PERIOD else 0
            volatility = np.std(close_prices[-RSI_PERIOD:]) if len(close_prices) >= RSI_PERIOD else 0

            print(f"[{datetime.now()}] {pair} price: ${latest_price:.2f}, RSI: {latest_rsi:.2f}, recent drop: {recent_drop:.2f}%, volatility: {volatility:.5f}")

            sold = simulate_sell(pair, latest_price, latest_rsi)

            if not sold and buys_this_cycle < max_buys_per_cycle:
                bought = simulate_buy(pair, latest_price, latest_rsi, close_prices)
                if bought:
                    buys_this_cycle += 1

            if pair in portfolio['positions']:
                amount = portfolio['positions'][pair]['amount']
                total_value += amount * latest_price

        print(f"[{datetime.now()}] Portfolio Value: ${total_value:.2f}\n")
        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()

