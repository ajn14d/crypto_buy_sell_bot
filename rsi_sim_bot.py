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

INTERVAL = 1  # in minutes (Kraken supports 1,5,15,30,60 etc.)
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
STARTING_BALANCE = 201.60
STOP_LOSS_PERCENT = 5.0        # Stop loss threshold %
RECENT_DROP_THRESHOLD = 5.0    # Max allowed price drop % in last 15 intervals before buying
LOOKBACK_PERIOD = 15           # Number of intervals to look back for recent drop
MAX_POSITIONS = 2              # Maximum simultaneous open positions
MIN_TRADE_USDT = 10            # Minimum USDT to use for a trade

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

# --- STOP LOSS CHECK & SELL ---
def check_stop_loss(pair, price):
    global portfolio
    if pair not in portfolio['positions']:
        return
    buy_price = portfolio['positions'][pair]['buy_price']
    loss_pct = ((price - buy_price) / buy_price) * 100
    if loss_pct <= -STOP_LOSS_PERCENT:
        amount = portfolio['positions'][pair]['amount']
        portfolio['USDT'] += amount * price
        print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
        del portfolio['positions'][pair]

# --- SIMULATION TRADE LOGIC ---
def sell_if_rsi_high(pair, price, rsi):
    global portfolio
    holding = pair in portfolio['positions']
    if holding and rsi > SELL_THRESHOLD:
        position = portfolio['positions'].pop(pair)
        value = position['amount'] * price
        portfolio['USDT'] += value
        profit = value - (position['amount'] * position['buy_price'])
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${profit:.2f}")

def main():
    print("Starting refined RSI simulation bot with best 2 buys per cycle...")
    while True:
        total_value = portfolio['USDT']

        buy_candidates = []

        # First pass: Fetch data, compute RSI, update price history, check stop loss & sells
        for pair in TRADING_PAIRS:
            df = fetch_ohlc(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                print(f"[{datetime.now()}] Not enough data for {pair}, skipping...")
                continue

            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            latest_rsi = df['rsi'].iloc[-1]
            latest_price = df['close'].iloc[-1]

            # Update price history
            history = price_history[pair]
            history.append(latest_price)
            if len(history) > LOOKBACK_PERIOD:
                history.pop(0)

            # Stop loss sell check
            check_stop_loss(pair, latest_price)

            # Sell if RSI high
            sell_if_rsi_high(pair, latest_price, latest_rsi)

            # Add position value to total
            if pair in portfolio['positions']:
                position = portfolio['positions'][pair]
                total_value += position['amount'] * latest_price

            # Collect buy candidates
            holding = pair in portfolio['positions']
            if not holding and latest_rsi < BUY_THRESHOLD:
                # Check recent drop
                if len(history) == LOOKBACK_PERIOD:
                    recent_drop = ((latest_price - history[0]) / history[0]) * 100
                else:
                    recent_drop = 0.0

                if recent_drop > -RECENT_DROP_THRESHOLD:
                    buy_candidates.append({
                        'pair': pair,
                        'price': latest_price,
                        'rsi': latest_rsi,
                        'recent_drop': recent_drop
                    })

        # Sort buy candidates by lowest RSI (strongest buy signals)
        buy_candidates.sort(key=lambda x: x['rsi'])

        # Calculate how many positions can be opened
        positions_available = MAX_POSITIONS - len(portfolio['positions'])
        if positions_available > 0 and len(buy_candidates) > 0:
            # Calculate USDT to spend per trade
            usdt_to_spend_per_trade = portfolio['USDT'] / positions_available
            if usdt_to_spend_per_trade < MIN_TRADE_USDT:
                print(f"[{datetime.now()}] Insufficient funds (${usdt_to_spend_per_trade:.2f}) to open new positions.")
            else:
                for candidate in buy_candidates[:positions_available]:
                    amount = usdt_to_spend_per_trade / candidate['price']
                    portfolio['positions'][candidate['pair']] = {
                        'amount': amount,
                        'buy_price': candidate['price']
                    }
                    portfolio['USDT'] -= usdt_to_spend_per_trade
                    print(f"[{datetime.now()}] BUY {candidate['pair']} @ ${candidate['price']:.2f} | "
                          f"RSI={candidate['rsi']:.2f} | Amount={amount:.6f} | Recent Drop={candidate['recent_drop']:.2f}%")

        print(f"[{datetime.now()}] Portfolio Value: ${total_value:.2f}\n")
        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()

