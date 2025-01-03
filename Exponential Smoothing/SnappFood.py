import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Load your data
monthly = pd.read_excel('food.xlsx', sheet_name='monthly')
daily = pd.read_excel('food.xlsx', sheet_name='daily')

# ------------------------monthly forecasting------------------------ #
df = pd.DataFrame(monthly)
df['YearMonth'] = df['Year'].astype(str) + '/' + df['Month'].astype(str).str.zfill(2)

df['Growth Rate'] = df['Monthly Order'].pct_change()*100
print(df[['YearMonth', 'Monthly Order', 'Growth Rate']])

monthly_seasonality = df.groupby('Month')['Monthly Order'].mean()
print(monthly_seasonality)

monthly_seasonality.plot(kind='bar', figsize=(10, 6), title='Average Monthly Orders: Seasonality')
plt.xlabel('Month')
plt.ylabel('Average Monthly Orders')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['YearMonth'], df['Monthly Order'], label='Monthly Orders', marker='o', color='skyblue')
plt.xlabel('Year/Month', fontsize=12)
plt.ylabel('Monthly Order', fontsize=12)
plt.title('Monthly Orders Over Time with Growth Rate', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.legend()
plt.show()

# Exponential Smoothing Model for Forecasting
train_data = df['Monthly Order']
model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
model_fit = model.fit()

# Forecasting for the next 3 months
forecast_months = 3
forecast = model_fit.forecast(forecast_months)

# Calculating MAE and RMSE on the historical data
y_true = df['Monthly Order'].iloc[-forecast_months:]
y_pred = forecast

mae = mean_absolute_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Adjusted projections for seasonality
last_month_orders = df['Monthly Order'].iloc[-1]
projected_orders = [last_month_orders * (1 + .10)**i for i in range(1, 4)]
adjusted_projections = []

for i, proj in enumerate(projected_orders):
    month = (df['Month'].iloc[-1] + i) % 12 + 1
    seasonal_factor = monthly_seasonality.loc[month] / monthly_seasonality.mean()
    adjusted_proj = proj * seasonal_factor
    adjusted_projections.append(adjusted_proj)

print("Adjusted Projected Orders with Seasonality:", adjusted_projections)

# Forecasting Visualization
historical_months = df['YearMonth'].tolist()
combined_months = historical_months + ['2024/07', '2024/08', '2024/09']
historical_orders = df['Monthly Order'].tolist()
combined_orders = historical_orders + list(forecast)

plt.figure(figsize=(12, 6))
plt.plot(combined_months, combined_orders, label='Monthly Orders (Forecasted)', marker='o', color='skyblue')
plt.xlabel('Year/Month', fontsize=12)
plt.ylabel('Monthly Order', fontsize=12)
plt.title('Monthly Orders Over Time with Forecast', fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.legend()
plt.show()

growth_percentage = [(adj - hist) /
                     hist * 100 for adj, hist in zip(adjusted_projections, df['Monthly Order'].iloc[-3:])]
print("Percentage Growth for the Next 3 Months:", growth_percentage)

# ------------------------daily forecasting------------------------ #
print(daily.info)

df = pd.DataFrame(daily)
df['YearMonthDay'] = (df['Year'].astype(str) + '/' + df['Month'].astype(str).str.zfill(2) +
                      '/' + df['Day'].astype(str).str.zfill(2))

peak_summary = df.groupby('PeakTime')['Daily Order'].agg(['sum', 'mean'])
print(peak_summary)

df['TotalDailyOrder'] = df.groupby(['Year', 'Month', 'Day'])['Daily Order'].transform(sum)
df['PeakContribution'] = np.where(
    df['PeakTime'] == 'PeakHour',
    (df['Daily Order'] / df['TotalDailyOrder']) * 100,
    0
)
print(df[['YearMonthDay', 'Daily Order', 'PeakTime', 'TotalDailyOrder', 'PeakContribution']].head())

daily_trends = df.groupby(['YearMonthDay'])['Daily Order'].sum()
daily_trends.plot(kind='line', figsize=(12, 6), title='Total Daily Orders Over Time')
plt.xlabel('Date')
plt.ylabel('Total Daily Orders')
plt.xticks(rotation=45)
plt.show()

weekday_avg_orders = df.groupby('Day')['Daily Order'].mean()
print(weekday_avg_orders)
weekday_avg_orders.plot(kind='bar', figsize=(10, 6), color='skyblue', title='Average Daily Orders by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Average Daily Orders')
plt.show()

pivot_data = df.pivot_table(index='YearMonthDay', columns='PeakTime', values='Daily Order', aggfunc='sum')
pivot_data.plot(kind='line', figsize=(12, 6))
plt.title('Daily Orders: Peak vs Non-Peak')
plt.ylabel('Daily Orders')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.legend(title='Time', labels=['Non-Peak', 'PeakHour'])
plt.tight_layout()
plt.show()

peak_order_percentage = (df[df['PeakTime'] == 'PeakHour'].groupby(['YearMonthDay'])['Daily Order'].sum() /
                         df.groupby(['YearMonthDay'])['Daily Order'].sum() * 100)
peak_order_percentage.plot(kind='line', figsize=(12, 6), title='Percentage of Peak Orders vs Total Orders')
plt.ylabel('Peak Order Percentage')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.show()

daily_growth_rate = df['Daily Order'].pct_change().mean()  # Average daily growth rate
next_90_days = pd.date_range(start='2024-07-01', end='2024-09-30')  # Adjust based on input
projected_daily_orders = [df['Daily Order'].iloc[-1] * (1 + daily_growth_rate) ** i for i in range(len(next_90_days))]

# -------------------------- Justification --------------------------
# - **Exponential Smoothing** was chosen for its ability to handle time series data effectively, especially in situations with trends and seasonality.
# - Exponential Smoothing is useful in capturing patterns from past observations, allowing the model to adjust and give more weight to recent data points, which is critical in business forecasting.
# - **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)** are used to validate the accuracy of the forecast. Lower values of these metrics indicate better performance of the model.
# - Exponential Smoothing is particularly suitable for data where trends and seasonality are prominent, which is the case with both monthly and daily order data in this scenario.
# - The model assumes that future data points will follow similar patterns as the historical data, making it a good fit for predicting future orders.
# - The method offers flexibility in terms of tuning the smoothing parameters to adjust the forecast’s sensitivity to recent observations.
# -------------------------------------------------------------------

# ------------------------budget planning------------------------ #
rider_fee = 4.00
peak_rider_fee = 4.00
fixed_operational_cost = 2000000
variable_operational_cost_per_order = 1.50

quarter_budget_target = 50000000

# Monthly Budget Calculation
def calculate_monthly_budget(month, monthly_order, peak_order_percentage):
    total_rider_fee = monthly_order * rider_fee
    if peak_order_percentage > 55:
        peak_order_count = monthly_order * (peak_order_percentage / 100)
        additional_peak_fee = peak_order_count * peak_rider_fee
    else:
        additional_peak_fee = 0
    total_rider_fee += additional_peak_fee
    total_operational_cost = fixed_operational_cost + (monthly_order * variable_operational_cost_per_order)
    total_monthly_cost = total_rider_fee + total_operational_cost
    cpo = total_monthly_cost / monthly_order

    return {
        'Month': month,
        'Monthly Order': monthly_order,
        'Peak Order Percentage': peak_order_percentage,
        'Total Rider Fee': total_rider_fee,
        'Total Operational Cost': total_operational_cost,
        'Total Monthly Cost': total_monthly_cost,
        'CPO': cpo
    }

# Apply the calculations for the forecasted months
monthly_budget = []
for i, proj in enumerate(adjusted_projections):
    month = ['2024/07', '2024/08', '2024/09'][i]
    peak_order_percentage_value = peak_order_percentage[i]  # Access the correct percentage for the month
    budget = calculate_monthly_budget(month, proj, peak_order_percentage_value)
    monthly_budget.append(budget)

# Calculate Quarterly Budget and CPO
total_quarter_cost = sum([budget['Total Monthly Cost'] for budget in monthly_budget])
total_quarter_orders = sum([budget['Monthly Order'] for budget in monthly_budget])
quarter_cpo = total_quarter_cost / total_quarter_orders

print("\nMonthly Budget Breakdown:")
for budget in monthly_budget:
    print(f"Month: {budget['Month']}, CPO: {budget['CPO']:.2f}, Total Cost: ${budget['Total Monthly Cost']:.2f}")

print(f"\nTotal Quarterly Cost: ${total_quarter_cost:.2f}")
print(f"Total Quarterly Orders: {total_quarter_orders}")
print(f"Quarter CPO: {quarter_cpo:.2f}")

# Recommendations for Budget Management
if total_quarter_cost > quarter_budget_target:
    print("\nWARNING: The forecasted demand exceeds the quarter budget target!")
    print("Recommendations:")
    print("1. Review peak hour order management to minimize additional peak fees.")
    print("2. Evaluate operational cost-saving opportunities, such as optimizing order fulfillment processes.")
    print("3. Consider adjusting pricing or discount strategies to manage demand.")
else:
    print("\nThe forecasted demand is within the quarter budget target. Continue monitoring closely.")
