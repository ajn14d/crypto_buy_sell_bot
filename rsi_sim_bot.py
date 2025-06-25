import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import csv
import os

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

INTERVAL = 1  # minutes (valid Kraken intervals: 1, 5, 15, etc.)
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
STARTING_BALANCE = 201.76
STOP_LOSS_PERCENT = 5.0
RECENT_DROP_THRESHOLD = 5.0
LOOKBACK_PERIOD = 15
TRADE_ALLOCATION = 0.5  # Use 50% of available USDT per buy
LOG_FILE = 'trade_log.csv'

# --- SIMULATED PORTFOLIO ---
portfolio = {
    'USDT': STARTING_BALANCE,
    'positions': {}  # {pair: {'amount': float, 'buy_price': float}}
}

price_history = {pair: [] for pair in TRADING_PAIRS}

# --- LOGGING SETUP ---
def init_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'action', 'pair', 'price', 'rsi', 'amount', 'pnl', 'portfolio_value'])

def log_trade(action, pair, price, rsi, amount, portfolio_value, pnl=None):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            action,
            pair,
            round(price, 4),
            round(rsi, 2),
            round(amount, 6),
            round(pnl, 2) if pnl is not None else '',
            round(portfolio_value, 2)
        ])

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

# --- FETCH OHLC DATA ---
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

# --- GET PORTFOLIO VALUE ---
def get_portfolio_value(current_prices=None):
    total = portfolio['USDT']
    for pair, pos in portfolio['positions'].items():
        if current_prices and pair in current_prices:
            total += pos['amount'] * current_prices[pair]
        else:
            df = fetch_ohlc(pair, INTERVAL)
            if df is not None:
                total += pos['amount'] * df['close'].iloc[-1]
    return total

# --- SIMULATED TRADE LOGIC ---
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

    # --- STOP LOSS ---
    if holding:
        buy_price = portfolio['positions'][pair]['buy_price']
        loss_pct = ((price - buy_price) / buy_price) * 100
        if loss_pct <= -STOP_LOSS_PERCENT:
            amount = portfolio['positions'][pair]['amount']
            portfolio['USDT'] += amount * price
            pnl = amount * (price - buy_price)
            value = get_portfolio_value({pair: price})
            print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
            log_trade('STOP LOSS', pair, price, rsi, amount, value, pnl)
            del portfolio['positions'][pair]
            return

    # --- RECENT DROP CHECK ---
    recent_drop = 0.0
    if len(history) == LOOKBACK_PERIOD:
        recent_drop = ((price - history[0]) / history[0]) * 100

    # --- BUY ---
    if rsi < BUY_THRESHOLD and not holding:
        if recent_drop > -RECENT_DROP_THRESHOLD:
            usdt_to_spend = portfolio['USDT'] * TRADE_ALLOCATION
            amount = usdt_to_spend / price
            if amount > 0:
                portfolio['positions'][pair] = {'amount': amount, 'buy_price': price}
                portfolio['USDT'] -= usdt_to_spend
                value = get_portfolio_value({pair: price})
                print(f"[{datetime.now()}] BUY {pair} @ ${price:.2f} | RSI={rsi:.2f} | Recent Drop={recent_drop:.2f}%")
                log_trade('BUY', pair, price, rsi, amount, value)
        else:
            print(f"[{datetime.now()}] Skipping {pair} due to recent drop: {recent_drop:.2f}%")

    # --- SELL ---
    elif rsi > SELL_THRESHOLD and holding:
        position = portfolio['positions'].pop(pair)
        value = position['amount'] * price
        portfolio['USDT'] += value
        pnl = value - (position['amount'] * position['buy_price'])
        port_val = get_portfolio_value({pair: price})
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${pnl:.2f}")
        log_trade('SELL', pair, price, rsi, position['amount'], port_val, pnl)

# --- MAIN LOOP ---
def main():
    init_logger()
    print("Starting RSI simulation bot with risk filters and trade logging...\n")
    while True:
        current_prices = {}
        for pair in TRADING_PAIRS:
            df = fetch_ohlc(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                continue
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            latest_rsi = df['rsi'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            current_prices[pair] = latest_price
            simulate_trade(pair, latest_price, latest_rsi)

        value = get_portfolio_value(current_prices)
        print(f"[{datetime.now()}] Portfolio Value: ${value:.2f}\n")
        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()

