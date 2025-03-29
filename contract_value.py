import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import sin, pi, atan2, sqrt

df = pd.read_csv("Nat_Gas.csv")
df.rename(columns={"Dates": "Date", "Prices": "Price"}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date").reset_index(drop=True)

start_date = df['Date'].iloc[0]
df['Days'] = (df['Date'] - start_date).dt.days
x = df['Days'].values
y = df['Price'].values

xbar = np.mean(x)
ybar = np.mean(y)
slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar)**2)
intercept = ybar - slope * xbar

residuals = y - (slope * x + intercept)
sin_component = np.sin(2 * pi * x / 365)
cos_component = np.cos(2 * pi * x / 365)
a = np.sum(residuals * sin_component) / np.sum(sin_component**2)
b = np.sum(residuals * cos_component) / np.sum(cos_component**2)
amplitude = sqrt(a**2 + b**2)
phase_shift = atan2(b, a)

def estimate_price(date_str):
    d = pd.to_datetime(date_str)
    days = (d - start_date).days
    return round(slope * days + intercept + amplitude * sin(2 * pi * days / 365 + phase_shift), 4)

def average_price_over_period(start_date_str, duration_days):
    start = pd.to_datetime(start_date_str)
    prices = []
    for offset in range(int(duration_days)):
        d = start + timedelta(days=offset)
        prices.append(estimate_price(d))
    return np.mean(prices)

def price_storage_contract(injection_dates, withdrawal_dates, volume_mmbtu,
                             injection_rate_mmbtu_per_day, withdrawal_rate_mmbtu_per_day,
                             max_storage_volume_mmbtu, storage_cost_per_month,
                             injection_withdrawal_cost_rate):
    assert len(injection_dates) == len(withdrawal_dates), "Mismatched injection/withdrawal dates"
    assert volume_mmbtu <= max_storage_volume_mmbtu, "Exceeds max storage capacity"
    total_revenue = 0
    total_cost = 0
    for inj_date, wd_date in zip(injection_dates, withdrawal_dates):
        inj_dt = pd.to_datetime(inj_date)
        wd_dt = pd.to_datetime(wd_date)
        inj_duration = volume_mmbtu / injection_rate_mmbtu_per_day
        wd_duration = volume_mmbtu / withdrawal_rate_mmbtu_per_day
        avg_buy_price = average_price_over_period(inj_date, inj_duration)
        avg_sell_price = average_price_over_period(wd_date, wd_duration)
        storage_start = inj_dt + timedelta(days=inj_duration)
        storage_duration_days = (wd_dt - storage_start).days
        storage_months = max(storage_duration_days / 30, 1)

        injection_cost = injection_withdrawal_cost_rate * volume_mmbtu
        withdrawal_cost = injection_withdrawal_cost_rate * volume_mmbtu

        revenue = avg_sell_price * volume_mmbtu
        cost = (
            avg_buy_price * volume_mmbtu +
            storage_cost_per_month * storage_months +
            injection_cost +
            withdrawal_cost
        )

        total_revenue += revenue
        total_cost += cost
    return round(total_revenue - total_cost, 2)

if __name__ == "__main__":
    injections = ['2024-06-30']
    withdrawals = ['2025-01-31']
    volume = 1_000_000
    contract_value = price_storage_contract(
        injections,
        withdrawals,
        volume,
        injection_rate_mmbtu_per_day=50_000,
        withdrawal_rate_mmbtu_per_day=50_000,
        max_storage_volume_mmbtu=2_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_rate=0.05
    )
    print("Contract Value:", contract_value)