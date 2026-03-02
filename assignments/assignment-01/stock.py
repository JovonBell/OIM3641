import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf

sb.set_theme()

"""
STUDENT CHANGE LOG & AI DISCLOSURE:
----------------------------------
1. Did you use an LLM (ChatGPT/Claude/etc.)? Yes
2. If yes, what was your primary prompt?
   Used Claude to help implement the Stock class methods
   including get_data, calc_returns, add_technical_indicators,
   plot_return_dist, and plot_performance.
----------------------------------
"""

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())


class Stock:
    def __init__(self, symbol="AAPL", start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol.upper()
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Downloads data from yfinance and triggers return calculation."""
        data = yf.download(self.symbol, start=self.start, end=self.end)
        if data.empty:
            raise ValueError(f"No data returned for {self.symbol}")
        # flatten MultiIndex columns yfinance returns for single tickers
        data.columns = [col[0] for col in data.columns]
        data.index = pd.to_datetime(data.index)
        self.calc_returns(data)
        return data

    def calc_returns(self, df):
        """Adds 'Change' and 'Instant_Return' columns to the dataframe."""
        df["Change"] = df["Close"].pct_change()
        df["Instant_Return"] = np.log(df["Close"]).diff().round(4)

    def add_technical_indicators(self, windows=[20, 50]):
        """Add SMAs and plot them with closing price."""
        for w in windows:
            self.data[f"SMA_{w}"] = self.data["Close"].rolling(window=w).mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data["Close"], label="Close")
        for w in windows:
            ax.plot(self.data.index, self.data[f"SMA_{w}"], label=f"SMA {w}")
        ax.set_title(f"{self.symbol} – Close with Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_return_dist(self):
        """Plots histogram of instantaneous returns."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.data["Instant_Return"].dropna(), bins=50, edgecolor="black")
        ax.set_title(f"{self.symbol} – Distribution of Instantaneous Returns")
        ax.set_xlabel("Instant Return")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        """Plots cumulative percent gain/loss over the period."""
        cumulative = (self.data["Close"] / self.data["Close"].iloc[0] - 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, cumulative)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f"{self.symbol} – Performance (% Gain/Loss)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        plt.tight_layout()
        plt.show()


def main():
    aapl = Stock("AAPL")
    print(aapl.data.head())
    aapl.plot_return_dist()
    aapl.plot_performance()
    aapl.add_technical_indicators()


if __name__ == "__main__":
    main()
