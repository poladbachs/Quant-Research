import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

df = pd.read_csv("Nat_Gas.csv")
df.rename(columns={"Dates": "Date", "Prices": "Price"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Price'], marker='o')
plt.title("Natural Gas Prices (Monthly Snapshot)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

df['Timestamp'] = df['Date'].map(datetime.toordinal)
X = df['Timestamp'].values.reshape(-1, 1)
y = df['Price'].values

model = LinearRegression()
model.fit(X, y)

def estimate_price(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = np.array([[date.toordinal()]])
    price = model.predict(timestamp)[0]
    return round(price, 4)

future_dates = pd.date_range(start=df['Date'].max() + pd.offsets.MonthEnd(1), periods=12, freq='M')
future_timestamps = future_dates.map(datetime.toordinal).values.reshape(-1, 1)
future_prices = model.predict(future_timestamps)

extrapolated_df = pd.DataFrame({
    'Date': future_dates,
    'Estimated_Price': np.round(future_prices, 4)
})

print("Extrapolated Natural Gas Prices (Next 12 Months):")
print(extrapolated_df)

print("Estimated price on 2025-04-15:", estimate_price("2025-04-15"))