"""
Bryan Neil Teddy Semester Project
Getting stock prices from API
Jan. 20 - Apr. 6
"""

"""
Methods of stock price comparison:
close - open
(high - low) / open
price at start date vs price today
"""

from datetime import datetime, date
import yfinance as yf

def get_stock(ticker):
    """
    :param ticker: str. name of stock ticker
    :return: DataFrame, data of stock price from Jan. 20, 2025 to today
    """
    data = yf.download(ticker, start = datetime(2025, 1, 20), end = datetime(2025, 4, 6))
    return data

def price_analysis(ticker, day):
    """
    :param ticker: str. name of stock ticker
    :param day: str. date ("YYYY-MM-DD") of day for prices to be analyzed
    :return: dict. with all volatility/price metrics included
    """
    data = get_stock(ticker)
    close = float(data.loc[day, "Close"].iloc[0])
    high = float(data.loc[day, "High"].iloc[0])
    low = float(data.loc[day, "Low"].iloc[0])
    open = float(data.loc[day, "Open"].iloc[0])
    dict = {"Close - Open:": close - open, "Day Volatility": (high-low)/open}
    return dict


if __name__ == "__main__":
    nasdaq = get_stock("VOO")
    print(nasdaq)
    print(type(nasdaq))
    print(price_analysis("VOO", "2025-03-20"))