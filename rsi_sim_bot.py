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

INTERVAL = 1  # in minutes
RSI_PERIOD = 14
BUY_THRESHOLD = 30
SELL_THRESHOLD = 70
STARTING_BALANCE = 202.00
STOP_LOSS_PERCENT = 5.0
RECENT_DROP_THRESHOLD = 5.0
LOOKBACK_PERIOD = 15

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

# --- SIMULATION LOGIC ---
def simulate_trade(pair, price, rsi, close_prices):
    global portfolio

    if rsi is None or rsi == 0 or np.isnan(rsi):
        print(f"[{datetime.now()}] Skipping {pair} due to invalid RSI: {rsi}")
        return False

    history = price_history[pair]
    history.append(price)
    if len(history) > LOOKBACK_PERIOD:
        history.pop(0)

    holding = pair in portfolio['positions']

    # Stop loss check (can sell anytime)
    if holding:
        buy_price = portfolio['positions'][pair]['buy_price']
        loss_pct = ((price - buy_price) / buy_price) * 100
        if loss_pct <= -STOP_LOSS_PERCENT:
            amount = portfolio['positions'][pair]['amount']
            portfolio['USDT'] += amount * price
            print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
            del portfolio['positions'][pair]
            return False  # This is a sell, not a buy

    if len(history) == LOOKBACK_PERIOD:
        recent_drop = ((price - history[0]) / history[0]) * 100
    else:
        recent_drop = 0.0

    volatility = np.std(close_prices[-RSI_PERIOD:]) if len(close_prices) >= RSI_PERIOD else 0

    if volatility > 0:
        allocation_pct = min(0.5, max(0.05, 0.2 / volatility))
    else:
        allocation_pct = 0.1

    # Buy logic - only buy if not holding and RSI < BUY_THRESHOLD
    if rsi < BUY_THRESHOLD and not holding:
        if recent_drop > -RECENT_DROP_THRESHOLD:
            amount_to_spend = portfolio['USDT'] * allocation_pct
            amount = amount_to_spend / price
            if amount > 0 and amount_to_spend <= portfolio['USDT']:
                portfolio['positions'][pair] = {'amount': amount, 'buy_price': price}
                portfolio['USDT'] -= amount_to_spend
                print(f"[{datetime.now()}] BUY {pair} @ ${price:.2f} | RSI={rsi:.2f} | Recent Drop={recent_drop:.2f}% | Allocated: ${amount_to_spend:.2f}")
                print(f"DEBUG after buy: USDT={portfolio['USDT']}, positions={portfolio['positions']}")
                return True  # buy happened
            else:
                print(f"[{datetime.now()}] Not enough USDT to buy {pair}")
                return False
        else:
            print(f"[{datetime.now()}] Skipping buy {pair} due to recent drop {recent_drop:.2f}%")
            return False

    # Sell logic - sell if holding and RSI > SELL_THRESHOLD
    elif rsi > SELL_THRESHOLD and holding:
        position = portfolio['positions'].pop(pair)
        value = position['amount'] * price
        portfolio['USDT'] += value
        profit = value - (position['amount'] * position['buy_price'])
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${profit:.2f}")
        return False

    return False

# --- MAIN LOOP ---
def main():
    print("Starting RSI simulation bot with volatility-based position sizing and debugging...")
    max_buys_per_cycle = 1

    while True:
        buys_this_cycle = 0
        total_value = portfolio['USDT']  # start with cash

        for pair in TRADING_PAIRS:
            df = fetch_ohlc(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                print(f"[{datetime.now()}] Not enough data for {pair}, skipping...")
                continue

            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            latest_rsi = df['rsi'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            close_prices = df['close'].values

            recent_drop = 0
            if len(close_prices) >= LOOKBACK_PERIOD:
                recent_drop = ((latest_price - close_prices[-LOOKBACK_PERIOD]) / close_prices[-LOOKBACK_PERIOD]) * 100
            volatility = np.std(close_prices[-RSI_PERIOD:]) if len(close_prices) >= RSI_PERIOD else 0

            print(f"[{datetime.now()}] {pair} price: ${latest_price:.2f}, RSI: {latest_rsi:.2f}, recent drop: {recent_drop:.2f}%, volatility: {volatility:.5f}")

            holding = pair in portfolio['positions']

            if buys_this_cycle < max_buys_per_cycle:
                bought = simulate_trade(pair, latest_price, latest_rsi, close_prices)
                if bought:
                    buys_this_cycle += 1
            else:
                # If no buys left, only run sell or stop-loss logic:
                # simulate_trade should be updated to separate buy vs sell, but here:
                # We call simulate_trade but inside it you can skip buying if buy limit reached
                # or separate simulate_sell function for clarity.
                # For now, you can do a minimal workaround:
                # Only allow selling/stop-loss if holding:
                if holding:
                    # Call simulate_trade but modify simulate_trade to allow selling even if buy limit reached
                    # Or create separate function for sell/stop loss logic.
                    # For now, just call simulate_trade:
                    simulate_trade(pair, latest_price, latest_rsi, close_prices)

            # Accumulate portfolio value by adding the value of holdings
            if holding:
                amount = portfolio['positions'][pair]['amount']
                total_value += amount * latest_price

        print(f"[{datetime.now()}] Portfolio Value: ${total_value:.2f}\n")
        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()