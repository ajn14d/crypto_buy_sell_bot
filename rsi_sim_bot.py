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

INTERVAL = 5  # in minutes
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
STARTING_BALANCE = 202.00
STOP_LOSS_PERCENT = 5.0  # Stop loss threshold %
RECENT_DROP_THRESHOLD = 5.0  # Max allowed price drop % in last 15 intervals before buying
LOOKBACK_PERIOD = 15  # Number of intervals to look back for recent drop

# --- VOLATILITY FILTER CONFIG ---
VOLATILITY_LOOKBACK = 10  # Number of candles to calculate volatility over
VOLATILITY_THRESHOLD = 1.5  # Minimum % price range to allow buying

# --- SIMULATED PORTFOLIO ---
portfolio = {
    'USDT': STARTING_BALANCE,
    'positions': {},  # e.g., {'ETHUSDT': {'amount': 0.1, 'buy_price': 1800}}
}

# --- PRICE HISTORY FOR RECENT DROP CHECK ---
price_history = {pair: [] for pair in TRADING_PAIRS}

# --- RSI CALCULATION ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- OHLC FETCH ---
def fetch_ohlc(pair, interval=15):
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {
        'pair': pair,
        'interval': interval
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if 'error' in data and data['error']:
            print(f"Error fetching {pair}: {data['error']}")
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
        print(f"Error fetching OHLC for {pair}: {e}")
        return None

# --- SIMULATION LOGIC ---
def simulate_trade(pair, price, rsi, close_prices):
    global portfolio

    if rsi is None or rsi == 0 or np.isnan(rsi):
        print(f"[{datetime.now()}] Skipping {pair} due to invalid RSI: {rsi}")
        return

    # Update price history for recent drop check
    history = price_history[pair]
    history.append(price)
    if len(history) > LOOKBACK_PERIOD:
        history.pop(0)

    holding = pair in portfolio['positions']

    # Check stop loss if holding
    if holding:
        buy_price = portfolio['positions'][pair]['buy_price']
        loss_pct = ((price - buy_price) / buy_price) * 100
        if loss_pct <= -STOP_LOSS_PERCENT:
            amount = portfolio['positions'][pair]['amount']
            portfolio['USDT'] += amount * price
            print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
            del portfolio['positions'][pair]
            return

    # Calculate recent drop if enough history
    if len(history) == LOOKBACK_PERIOD:
        recent_drop = ((price - history[0]) / history[0]) * 100
    else:
        recent_drop = 0.0  # not enough data yet

    # Calculate volatility over last VOLATILITY_LOOKBACK candles
    if len(close_prices) >= VOLATILITY_LOOKBACK:
        recent_closes = close_prices[-VOLATILITY_LOOKBACK:]
        volatility = (max(recent_closes) - min(recent_closes)) / min(recent_closes) * 100
    else:
        volatility = 0.0

    # Buy logic with volatility filter
    if rsi < BUY_THRESHOLD and not holding:
        if recent_drop > -RECENT_DROP_THRESHOLD:
            if volatility >= VOLATILITY_THRESHOLD:
                usdt = portfolio['USDT']
                amount = usdt / price
                if amount > 0:
                    portfolio['positions'][pair] = {'amount': amount, 'buy_price': price}
                    portfolio['USDT'] = 0
                    print(f"[{datetime.now()}] BUY {pair} @ ${price:.2f} | RSI={rsi:.2f} | Recent Drop={recent_drop:.2f}% | Volatility={volatility:.2f}%")
            else:
                print(f"[{datetime.now()}] Skipping buy {pair} due to low volatility: {volatility:.2f}%")
        else:
            print(f"[{datetime.now()}] Skipping buy {pair} due to recent drop {recent_drop:.2f}%")

    # Sell logic
    elif rsi > SELL_THRESHOLD and holding:
        position = portfolio['positions'].pop(pair)
        value = position['amount'] * price
        portfolio['USDT'] += value
        profit = value - (position['amount'] * position['buy_price'])
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${profit:.2f}")

# --- MAIN LOOP ---
def main():
    print("Starting RSI simulation bot with volatility filter...")
    while True:
        total_value = portfolio['USDT']
        for pair in TRADING_PAIRS:
            df = fetch_ohlc(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                print(f"[{datetime.now()}] Not enough data for {pair}, skipping...")
                continue

            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            latest_rsi = df['rsi'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            close_prices = df['close'].values

            simulate_trade(pair, latest_price, latest_rsi, close_prices)

            # Calculate portfolio value including open positions
            if pair in portfolio['positions']:
                position = portfolio['positions'][pair]
                total_value += position['amount'] * latest_price

        print(f"[{datetime.now()}] Portfolio Value: ${total_value:.2f}\n")

        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()


