import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import robin_stocks.robinhood as r
import pyotp
import time
from datetime import datetime, timedelta

class MLAdaptiveSuperTrend:
    def __init__(self, symbol, atr_period=14, factor=3, training_period=500):
        self.symbol = symbol
        self.atr_period = atr_period
        self.factor = factor
        self.training_period = training_period
        self.volatility_levels = None
        
    def fetch_data(self, interval='day', span='year'):
        historical_data = r.stocks.get_stock_historicals(self.symbol, interval=interval, span=span)
        df = pd.DataFrame(historical_data)
        df['datetime'] = pd.to_datetime(df['begins_at'])
        df.set_index('datetime', inplace=True)
        df['close'] = df['close_price'].astype(float)
        df['high'] = df['high_price'].astype(float)
        df['low'] = df['low_price'].astype(float)
        return df
    
    def calculate_atr(self, df):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(self.atr_period).mean()
    
    def cluster_volatility(self, atr_values):
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(atr_values.reshape(-1, 1))
        self.volatility_levels = np.sort(kmeans.cluster_centers_.flatten())
    
    def calculate_supertrend(self, df):
        hl2 = (df['high'] + df['low']) / 2
        atr = self.calculate_atr(df)
        
        if self.volatility_levels is None:
            self.cluster_volatility(atr[:self.training_period].values)
        
        volatility_level = pd.cut(atr, bins=[-np.inf] + list(self.volatility_levels) + [np.inf], labels=[0, 1, 2])
        factor = self.factor * (1 + volatility_level * 0.5)  # Adjust factor based on volatility
        
        upperband = hl2 + (factor * atr)
        lowerband = hl2 - (factor * atr)
        
        supertrend = pd.Series(index=df.index)
        direction = pd.Series(index=df.index)
        
        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == upperband.iloc[i-1] and df['close'].iloc[i] <= upperband.iloc[i]:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == upperband.iloc[i-1] and df['close'].iloc[i] > upperband.iloc[i]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = -1
            elif supertrend.iloc[i-1] == lowerband.iloc[i-1] and df['close'].iloc[i] >= lowerband.iloc[i]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = -1
            elif supertrend.iloc[i-1] == lowerband.iloc[i-1] and df['close'].iloc[i] < lowerband.iloc[i]:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = 1
        
        return supertrend, direction

    def generate_signals(self, df):
        supertrend, direction = self.calculate_supertrend(df)
        df['supertrend'] = supertrend
        df['direction'] = direction
        df['signal'] = direction.diff()
        return df

class RobinhoodTrader:
    def __init__(self, username, password, totp_key):
        self.username = username
        self.password = password
        self.totp_key = totp_key
        self.login()
    
    def login(self):
        totp = pyotp.TOTP(self.totp_key).now()
        r.login(self.username, self.password, mfa_code=totp)
    
    def get_buying_power(self):
        profile = r.profiles.load_account_profile()
        return float(profile['buying_power'])
    
    def place_order(self, symbol, quantity, side):
        try:
            if side == 'buy':
                r.orders.order_buy_market(symbol, quantity)
                print(f"Bought {quantity} shares of {symbol}")
            elif side == 'sell':
                r.orders.order_sell_market(symbol, quantity)
                print(f"Sold {quantity} shares of {symbol}")
        except Exception as e:
            print(f"Error placing order: {e}")

def run_strategy(symbol, username, password, totp_key):
    trader = RobinhoodTrader(username, password, totp_key)
    strategy = MLAdaptiveSuperTrend(symbol)
    
    while True:
        try:
            df = strategy.fetch_data()
            signals = strategy.generate_signals(df)
            
            last_signal = signals.iloc[-1]
            
            if last_signal['signal'] == 2:  # Buy signal
                buying_power = trader.get_buying_power()
                current_price = float(last_signal['close'])
                quantity = int(buying_power * 0.1 / current_price)  # Use 10% of buying power
                if quantity > 0:
                    trader.place_order(symbol, quantity, 'buy')
            elif last_signal['signal'] == -2:  # Sell signal
                positions = r.account.get_open_stock_positions()
                for position in positions:
                    if position['symbol'] == symbol:
                        quantity = int(float(position['quantity']))
                        if quantity > 0:
                            trader.place_order(symbol, quantity, 'sell')
            
            print(f"Last signal for {symbol}: {last_signal['signal']} at {last_signal.name}")
            time.sleep(60 * 60)  # Wait for 1 hour before next check
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)  # Wait for 1 minute before retrying

if __name__ == "__main__":
    symbol = "AAPL"  # Example stock
    username = "your_robinhood_username"
    password = "your_robinhood_password"
    totp_key = "your_totp_key"
    
    run_strategy(symbol, username, password, totp_key)
