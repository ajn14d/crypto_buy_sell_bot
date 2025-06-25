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
STARTING_BALANCE = 202.10
STOP_LOSS_PERCENT = 5.0
RECENT_DROP_THRESHOLD = 5.0
LOOKBACK_PERIOD = 15
MIN_TRADE_SIZE = 50  # minimum USD amount for any buy or sell

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
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Fix RSI when avg_loss == 0 (means rs = inf), set RSI = 100
    rsi[avg_loss == 0] = 100

    # Fix RSI when avg_gain == 0 and avg_loss == 0 (flat line), RSI = 50 neutral
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50

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

# --- BUY LOGIC ---
def simulate_buy(pair, price, rsi, close_prices):
    global portfolio

    if rsi is None or rsi == 0 or np.isnan(rsi):
        print(f"[{datetime.now()}] Skipping buy {pair} due to invalid RSI: {rsi}")
        return False

    holding = pair in portfolio['positions']
    if holding:
        return False  # Already holding, no buy

    history = price_history[pair]
    history.append(price)
    if len(history) > LOOKBACK_PERIOD:
        history.pop(0)

    if len(history) == LOOKBACK_PERIOD:
        recent_drop = ((price - history[0]) / history[0]) * 100
    else:
        recent_drop = 0.0

    if recent_drop <= -RECENT_DROP_THRESHOLD:
        print(f"[{datetime.now()}] Skipping buy {pair} due to recent drop {recent_drop:.2f}%")
        return False

    volatility = np.std(close_prices[-RSI_PERIOD:]) if len(close_prices) >= RSI_PERIOD else 0
    if volatility > 0:
        allocation_pct = min(0.5, max(0.05, 0.2 / volatility))
    else:
        allocation_pct = 0.1

    if rsi < BUY_THRESHOLD:
        amount_to_spend = portfolio['USDT'] * allocation_pct
        
        # --- New: Minimum trade size check ---
        if amount_to_spend < MIN_TRADE_SIZE:
            print(f"[{datetime.now()}] Skipping buy {pair} because allocation ${amount_to_spend:.2f} < minimum trade size ${MIN_TRADE_SIZE}")
            return False

        amount = amount_to_spend / price
        if amount > 0 and amount_to_spend <= portfolio['USDT']:
            portfolio['positions'][pair] = {'amount': amount, 'buy_price': price}
            portfolio['USDT'] -= amount_to_spend
            print(f"[{datetime.now()}] BUY {pair} @ ${price:.2f} | RSI={rsi:.2f} | Recent Drop={recent_drop:.2f}% | Allocated: ${amount_to_spend:.2f}")
            print(f"DEBUG after buy: USDT={portfolio['USDT']}, positions={portfolio['positions']}")
            return True
        else:
            print(f"[{datetime.now()}] Not enough USDT to buy {pair}")
    return False

# --- SELL LOGIC (includes stop loss) ---
def simulate_sell(pair, price, rsi):
    global portfolio

    holding = pair in portfolio['positions']
    if not holding:
        return False  # Nothing to sell

    buy_price = portfolio['positions'][pair]['buy_price']
    amount = portfolio['positions'][pair]['amount']
    loss_pct = ((price - buy_price) / buy_price) * 100
    trade_value = amount * price

    # --- New: Minimum trade size check ---
    if trade_value < MIN_TRADE_SIZE:
        print(f"[{datetime.now()}] Skipping sell {pair} because trade value ${trade_value:.2f} < minimum trade size ${MIN_TRADE_SIZE}")
        return False

    # Stop loss triggered
    if loss_pct <= -STOP_LOSS_PERCENT:
        portfolio['USDT'] += trade_value
        print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
        del portfolio['positions'][pair]
        return True

    # Normal sell if RSI above threshold
    if rsi > SELL_THRESHOLD:
        portfolio['USDT'] += trade_value
        profit = (price - buy_price) * amount
        print(f"[{datetime.now()}] SELL {pair} @ ${price:.2f} | RSI={rsi:.2f} | PnL=${profit:.2f}")
        del portfolio['positions'][pair]
        return True

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

            # Always check for sell first (or stop loss)
            sold = simulate_sell(pair, latest_price, latest_rsi)

            # If not sold, try buy if limit not reached
            if not sold and buys_this_cycle < max_buys_per_cycle:
                bought = simulate_buy(pair, latest_price, latest_rsi, close_prices)
                if bought:
                    buys_this_cycle += 1

            # Calculate portfolio value safely
            if pair in portfolio['positions']:
                amount = portfolio['positions'][pair]['amount']
                total_value += amount * latest_price

        print(f"[{datetime.now()}] Portfolio Value: ${total_value:.2f}\n")
        time.sleep(INTERVAL * 60)

if __name__ == '__main__':
    main()
