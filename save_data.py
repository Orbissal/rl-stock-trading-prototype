import yfinance as yf
import pandas as pd

tickers = {
    'AAPL': ('2018-01-01', '2024-11-01'),
    'MSFT': ('2018-01-01', '2024-11-01'),
    'GOOGL': ('2018-01-01', '2024-11-01'),
}

for ticker, (start, end) in tickers.items():
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.to_csv(f'data_{ticker}.csv')
    print(f"Saved data_{ticker}.csv — {len(data)} days")

print("\nAll data saved. You can now work offline.")


