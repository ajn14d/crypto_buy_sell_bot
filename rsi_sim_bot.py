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

KRAKEN_PAIRS = {
    'ETHUSDT': 'ETHUSD',
    'BTCUSDT': 'XBTUSD',
    'SOLUSDT': 'SOLUSD',
    'AVAXUSDT': 'AVAXUSD',
    'DOGEUSDT': 'DOGEUSD',
    'ADAUSDT': 'ADAUSD',
    'LTCUSDT': 'LTCUSD',
    'XRPUSDT': 'XRPUSD'
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

    rsi[avg_loss == 0] = 100
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50

    return rsi

# --- OHLC FETCH FROM KRAKEN ---
def fetch_ohlc_kraken(pair, interval=1):
    url = 'https://api.kraken.com/0/public/OHLC'
    params = {'pair': pair, 'interval': interval}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if 'error' in data and data['error']:
            print(f"Error fetching {pair} from Kraken: {data['error']}")
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
        print(f"Error fetching OHLC for {pair} from Kraken: {e}")
        return None

# --- OHLC FETCH FROM BINANCE (Fallback) ---
def fetch_ohlc_binance(pair, interval=1):
    # Binance uses lowercase symbols and intervals like '1m'
    binance_pair = pair.lower()
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
    kraken_pair = KRAKEN_PAIRS.get(pair, pair)  # Map to Kraken symbol if available
    for attempt in range(1, MAX_RETRIES + 1):
        df = fetch_ohlc_kraken(kraken_pair, interval)
        if df is not None:
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            latest_rsi = df['rsi'].iloc[-1]
            if latest_rsi is not None and latest_rsi > 0:
                return df
            else:
                print(f"[{datetime.now()}] Attempt {attempt}: Invalid RSI for {pair}: {latest_rsi}")
        else:
            print(f"[{datetime.now()}] Attempt {attempt}: Failed to fetch data for {pair} from Kraken")

        time.sleep(1)  # short delay before retry

    # Fallback to Binance
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
        # Only buy if RSI is below buy threshold
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

    # --- CHANGE HERE: Calculate volatility as std of pct returns ---
    returns = pd.Series(close_prices).pct_change().dropna()
    volatility = returns[-RSI_PERIOD:].std() if len(returns) >= RSI_PERIOD else 0

    if volatility > 0:
        allocation_pct = min(0.5, max(0.05, 0.2 / volatility))
    else:
        allocation_pct = 0.1

    amount_to_spend = portfolio['USDT'] * allocation_pct

    if amount_to_spend < MIN_TRADE_USD:
        print(f"[{datetime.now()}] Skipping buy {pair} because allocation ${amount_to_spend:.2f} < minimum trade size ${MIN_TRADE_USD}")
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

    # Stop loss triggered
    if loss_pct <= -STOP_LOSS_PERCENT:
        portfolio['USDT'] += amount * price
        print(f"[{datetime.now()}] [STOP LOSS] SELL {pair} @ ${price:.2f} | Loss: {loss_pct:.2f}%")
        del portfolio['positions'][pair]
        return True

    # Normal sell if RSI above threshold
    if rsi > SELL_THRESHOLD:
        portfolio['USDT'] += amount * price
        profit = (price - buy_price) * amount
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
        total_value = portfolio['USDT']  # start with cash

        for pair in TRADING_PAIRS:
            df = fetch_data_with_retry(pair, INTERVAL)
            if df is None or len(df) < RSI_PERIOD:
                print(f"[{datetime.now()}] Not enough data for {pair}, skipping...")
                continue

            latest_rsi = df['rsi'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            close_prices = df['close'].values

            recent_drop = 0
            if len(close_prices) >= LOOKBACK_PERIOD:
                recent_drop = ((latest_price - close_prices[-LOOKBACK_PERIOD]) / close_prices[-LOOKBACK_PERIOD]) * 100

            # --- CHANGE HERE: Calculate volatility as std of pct returns ---
            returns = pd.Series(close_prices).pct_change().dropna()
            volatility = returns[-RSI_PERIOD:].std() if len(returns) >= RSI_PERIOD else 0

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


