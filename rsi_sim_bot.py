import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import csv
import os

# --- CONFIGURATION ---
TRADING_PAIRS = [
    'ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'AVAXUSDT',
    'DOGEUSDT', 'ADAUSDT', 'LTCUSDT', 'XRPUSDT'
]

INTERVAL = 1  # in minutes
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
STARTING_BALANCE = 201.75
STOP_LOSS_PERCENT = 5.0
RECENT_DROP_THRESHOLD = 5.0
LOOKBACK_PERIOD = 15
MAX_POSITIONS = 2
TRADE_PERCENT = 0.5
LOG_FILE = 'trade_log.csv'

# --- SIMULATED PORTFOLIO ---
portfolio = {
    'USDT': STARTING_BALANCE,
    'positions': {}
}
price_history = {pair: [] for pair in TRADING_PAIRS}

# --- CSV LOGGER SETUP ---
def init_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'action', 'pair', 'price', 'rsi', 'amount', 'pnl'])

def log_trade(action, pair, price, rsi, amount, pnl=None):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            action,
            pair,
            round(price, 4),
            round(rsi, 2),
            round(amount, 6),
            round(pnl, 2) if pnl is not None else ''
        ])

# --- RSI CALCULATION ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- OHLC FETCH ---
def fetch_ohlc(pair, interval=15):
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {'pair': pair, 'interval': interval}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if 'error' in data and data['error']:
            print(f"Error fetching {pair}: {data['error']}")
            return None
        key = list(data['result'].keys())[0]
        df = pd.DataFrame(data['result'][key], columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching OHLC for {pair}: {e}")
        return None

# --- SIMULATION LOGIC ---
def simulate_trade(pair, price, rsi):
    global portfolio

    if rsi is None or rsi == 0 or np.isnan(rsi):
        print(f"[{datetime.now()}] RSI not valid for {pair}, skipping...")
        return

    history = price_history[pair]
    history.append(price)
    if len(history) > LOOKBACK_PERIOD:
        history.pop(0)

    holding = pair in portfolio['positions']

    # STOP LOSS
    if holding:
        buy_price = portfolio['positions'][pair]['buy_price']
        amount = portfolio['positions'][pair]['amount']
        loss_pct = ((price - buy_price) / buy_price) * 100
        if loss_pct <= -STOP_LOSS_PERCENT:
            portfolio['USDT'] += amount * price
            del portfolio['positions'][pair]
            print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
            log_trade('STOP LOSS', pair, price, rsi, amount, pnl=(amount * (price - buy_price)))
            return

    # RECENT DROP
    if len(history) == LOOKBACK_PERIOD:
        recent_drop = ((price - history[0]) / history[0]) * 100
    else:
        recent_drop = 0.0

    # BUY LOGIC
    if rsi < BUY_THRESHOLD and not holding and len(portfolio['positions']) < MAX_POSITIONS:
        if recent_drop > -RECENT_DROP_THRESHOLD:
            usdt_to_use = portfolio['USDT'] * TRADE_PERCENT
            amount = usdt_to_use / price
            if amount > 0:
                portfolio['positions'][pair] = {'amount': amount, 'buy_price': price}
                portfolio['USDT'] -= usdt_to_use
                print(f"[{datetime.now()}] BUY {pair} @ ${price:.2f} | RSI={rsi:.2f} | Recent Drop={recent_drop:.2f}%")
                log_trade('BUY', pair, price, rsi, amount)
        else:
            print(f"[{datetime.now()}] Skipping buy {pair} due to recent drop {recent_drop:.2f}%")

    # SELL LOGIC
    elif rsi > SELL_THRESHOLD and holding:
        position = portfolio['positions'].pop(pair)
        value = position['amount'] * price
        profit = value - (position['amount'] * position['buy_price'])
        portfolio['USDT'] += value
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${profit:.2f}")
        log_trade('SELL', pair, price, rsi, position['amount'], pnl=profit)

# --- MAIN LOOP ---
def main():
    init_logger()
    print("Starting RSI bot with logging...")
    while True:
        total_value = portfolio['USDT']
        candidates = []

        for pair in TRADING_PAIRS:
            df = fetch_ohlc(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                continue

            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            rsi = df['rsi'].iloc[-1]
            price = df['close'].iloc[-1]

            candidates.append((pair, price, rsi))

        # Sort by lowest RSI if buying
        if len(portfolio['positions']) < MAX_POSITIONS:
            sorted_candidates = sorted(
                [c for c in candidates if c[2] < BUY_THRESHOLD],
                key=lambda x: x[2]
            )
            for pair, price, rsi in sorted_candidates:
                simulate_trade(pair, price, rsi)
                if len(portfolio['positions']) >= MAX_POSITIONS:
                    break

        # Evaluate all for selling or stop-loss
        for pair, price, rsi in candidates:
            simulate_trade(pair, price, rsi)

        # Portfolio value printout
        total_value = portfolio['USDT']
        for pair, position in portfolio['positions'].items():
            df = fetch_ohlc(pair, INTERVAL)
            if df is not None:
                price = df['close'].iloc[-1]
                total_value += position['amount'] * price

        print(f"[{datetime.now()}] Portfolio Value: ${total_value:.2f}\n")
        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()

