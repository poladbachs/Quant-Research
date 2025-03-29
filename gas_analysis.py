import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

df = pd.read_csv("Nat_Gas.csv")
df.rename(columns={"Dates": "Date", "Prices": "Price"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

start_date = date(2020, 10, 31)
end_date = date(2024, 9, 30)
months = []
year = start_date.year
month = start_date.month + 1
while True:
    current = date(year, month, 1) + timedelta(days=-1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    else:
        month = ((month + 1) % 12) or 12
        if month == 1:
            year += 1

days_from_start = [(day - start_date).days for day in months]
prices = df['Price'].values
time = np.array(days_from_start)

xbar = np.mean(time)
ybar = np.mean(prices)
slope = np.sum((time - xbar) * (prices - ybar)) / np.sum((time - xbar) ** 2)
intercept = ybar - slope * xbar

sin_prices = prices - (time * slope + intercept)
sin_time = np.sin(time * 2 * np.pi / 365)
cos_time = np.cos(time * 2 * np.pi / 365)
slope1 = np.sum(sin_prices * sin_time) / np.sum(sin_time ** 2)
slope2 = np.sum(sin_prices * cos_time) / np.sum(cos_time ** 2)

amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift = np.arctan2(slope2, slope1)

def estimate_price(date_str):
    d = pd.to_datetime(date_str).date()
    days = (d - start_date).days
    return round(amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept, 4)

future_dates = pd.date_range(start=df['Date'].max() + pd.offsets.MonthEnd(1), periods=12, freq='M')
future_days = [(d.date() - start_date).days for d in future_dates]
future_prices = [estimate_price(str(d.date())) for d in future_dates]

extrapolated_df = pd.DataFrame({
    'Date': future_dates,
    'Estimated_Price': future_prices
})

print(extrapolated_df)
print("Estimated price on 2025-04-15:", estimate_price("2025-04-15"))