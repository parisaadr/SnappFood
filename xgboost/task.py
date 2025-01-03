import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


monthly = pd.read_excel('food.xlsx', sheet_name='monthly')
daily = pd.read_excel('food.xlsx', sheet_name='daily')

# -------------------------- Monthly Forecast --------------------------

monthly['Date'] = pd.to_datetime(monthly[['Year', 'Month']].assign(DAY=1))
monthly['YearMonth'] = monthly['Date'].dt.strftime('%Y/%m')
monthly['Lag1'] = monthly['Monthly Order'].shift(1)
monthly['Lag2'] = monthly['Monthly Order'].shift(2)
monthly['Lag3'] = monthly['Monthly Order'].shift(3)
monthly['GrowthRate'] = monthly['Monthly Order'].pct_change() * 100
monthly['Month'] = monthly['Date'].dt.month

monthly = monthly.dropna()

# Model Training
X_monthly = monthly[['Lag1', 'Lag2', 'Lag3', 'GrowthRate', 'Month']]
y_monthly = monthly['Monthly Order']

X_train_monthly, X_test_monthly, y_train_monthly, y_test_monthly = train_test_split(
    X_monthly, y_monthly, test_size=0.2, shuffle=False
)

model_monthly = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
model_monthly.fit(X_train_monthly, y_train_monthly)

y_pred_monthly = model_monthly.predict(X_test_monthly)

# Evaluate
rmse_monthly = np.sqrt(mean_squared_error(y_test_monthly, y_pred_monthly))
print(f'Monthly RMSE: {rmse_monthly}')

# Visualize Monthly Forecast
plt.figure(figsize=(12, 6))
plt.plot(y_test_monthly.index, y_test_monthly, label='Actual Orders', color='blue')
plt.plot(y_test_monthly.index, y_pred_monthly, label='Predicted Orders', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Monthly Orders')
plt.title('Monthly Orders: Forecast vs Actual')
plt.legend()
plt.show()

# Compare with Moving Average (Simpler Model)
moving_avg = monthly['Monthly Order'].rolling(window=3).mean()
moving_avg_rmse = np.sqrt(mean_squared_error(monthly['Monthly Order'].iloc[2:], moving_avg.dropna()))
print(f'Moving Average RMSE: {moving_avg_rmse}')

# -------------------------- Daily Forecast --------------------------

# Feature Engineering
daily['Date'] = pd.to_datetime(daily[['Year', 'Month', 'Day']])
daily_pivot = daily.pivot_table(index='Date', columns='PeakTime', values='Daily Order')
daily_pivot.columns = ['NonPeak', 'PeakHour']
daily_pivot = daily_pivot.reset_index()
daily_pivot['DayOfWeek'] = daily_pivot['Date'].dt.dayofweek

# Ensure 'Date' is the index of the dataframe and set frequency
daily_pivot.set_index('Date', inplace=True)
daily_pivot = daily_pivot.asfreq('D')  # Set frequency to daily (or 'H' for hourly if needed)

# Add Seasonal Decomposition (ensure proper frequency is set)
stl = STL(daily_pivot['PeakHour'], seasonal=7)
stl_result = stl.fit()
daily_pivot['PeakHour_Seasonal'] = stl_result.seasonal

# Lag Features
for lag in range(1, 4):
    daily_pivot[f'PeakHour_Lag{lag}'] = daily_pivot['PeakHour'].shift(lag)
    daily_pivot[f'NonPeak_Lag{lag}'] = daily_pivot['NonPeak'].shift(lag)

# Rolling Averages
daily_pivot['PeakHour_Roll7'] = daily_pivot['PeakHour'].rolling(window=7).mean()
daily_pivot['NonPeak_Roll7'] = daily_pivot['NonPeak'].rolling(window=7).mean()

daily_pivot = daily_pivot.dropna()

# Model Training
X_daily = daily_pivot.drop(columns=['PeakHour', 'NonPeak'])
y_peak = daily_pivot['PeakHour']
y_nonpeak = daily_pivot['NonPeak']

X_train_daily, X_test_daily, y_train_peak, y_test_peak = train_test_split(
    X_daily, y_peak, test_size=0.2, shuffle=False
)
_, _, y_train_nonpeak, y_test_nonpeak = train_test_split(X_daily, y_nonpeak, test_size=0.2, shuffle=False)

model_peak = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
model_nonpeak = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)

model_peak.fit(X_train_daily, y_train_peak)
model_nonpeak.fit(X_train_daily, y_train_nonpeak)

y_pred_peak = model_peak.predict(X_test_daily)
y_pred_nonpeak = model_nonpeak.predict(X_test_daily)

# Evaluate
rmse_peak = np.sqrt(mean_squared_error(y_test_peak, y_pred_peak))
rmse_nonpeak = np.sqrt(mean_squared_error(y_test_nonpeak, y_pred_nonpeak))
print(f'PeakHour RMSE: {rmse_peak}')
print(f'NonPeak RMSE: {rmse_nonpeak}')

# Visualize Daily Forecast
plt.figure(figsize=(12, 6))
plt.plot(y_test_peak.index, y_test_peak, label='Actual PeakHour', color='blue')
plt.plot(y_test_peak.index, y_pred_peak, label='Predicted PeakHour', color='red', linestyle='dashed')
plt.title('Daily PeakHour: Forecast vs Actual')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_nonpeak.index, y_test_nonpeak, label='Actual NonPeak', color='green')
plt.plot(y_test_nonpeak.index, y_pred_nonpeak, label='Predicted NonPeak', color='orange', linestyle='dashed')
plt.title('Daily NonPeak: Forecast vs Actual')
plt.legend()
plt.show()

# -------------------------- Justification --------------------------
# - XGBoost was chosen for its ability to handle nonlinear relationships and its feature flexibility.
# - Seasonal decomposition was added to explicitly account for daily trends.
# - Moving Average model serves as a baseline comparison for simplicity.
# - Assumptions: Future patterns align with historical data, and lag features capture enough seasonality.
# -------------------------------------------------------------------


# Monthly Order Forecast from the previous task (y_pred_monthly is the predicted monthly order)
monthly_orders = y_pred_monthly  # Replace with actual predicted monthly order data from your model
total_months = len(monthly_orders)

# Initialize cost variables
rider_fee_per_order = 4.00
peak_time_extra_fee = 4.00
peak_threshold = 0.55  # 55% peak time orders threshold
fixed_operational_costs = 2000000  # $2,000,000 fixed overhead per month
variable_cost_per_order = 1.50  # $1.50 per order variable cost

# Calculate the total budget costs for each month
monthly_budget = []

for month in range(total_months):
    # Assume we have the forecasted peak orders and total orders for the month
    # You would have to sum up your daily peak and non-peak orders for each month
    total_orders = monthly_orders[month]

    # Calculate Rider Fees
    rider_fees = total_orders * rider_fee_per_order

    # Assume a random daily peak rate (this should come from your actual forecasted data)
    # For demonstration, let's assume 60% of the orders are during peak hours (modify with your actual data)
    peak_share = 0.60  # Example percentage of peak time orders

    peak_time_rider_fees = 0
    if peak_share > peak_threshold:
        peak_time_orders = total_orders * peak_share
        peak_time_rider_fees = (peak_time_orders - total_orders * peak_threshold) * peak_time_extra_fee

    # Calculate Operational Costs
    operational_costs = fixed_operational_costs + (total_orders * variable_cost_per_order)

    # Calculate Total Cost
    total_cost = rider_fees + peak_time_rider_fees + operational_costs

    # Calculate CPO (Cost Per Order)
    cpo = total_cost / total_orders

    # Store monthly budget data
    monthly_budget.append({
        'Month': month + 1,
        'Total Orders': total_orders,
        'Rider Fees': rider_fees,
        'Peak Time Rider Fees': peak_time_rider_fees,
        'Operational Costs': operational_costs,
        'Total Cost': total_cost,
        'CPO': cpo
    })

# Convert the budget data into a DataFrame
budget_df = pd.DataFrame(monthly_budget)

# Visualize the Budget Breakdown
import matplotlib.pyplot as plt

# Plot Monthly Budget
plt.figure(figsize=(12, 6))
plt.bar(budget_df['Month'], budget_df['Total Cost'], label='Total Cost', color='lightblue')
plt.bar(budget_df['Month'], budget_df['Rider Fees'], label='Rider Fees', color='blue', alpha=0.7)
plt.bar(budget_df['Month'], budget_df['Peak Time Rider Fees'], label='Peak Time Rider Fees', color='orange', alpha=0.7)
plt.bar(budget_df['Month'], budget_df['Operational Costs'], label='Operational Costs', color='green', alpha=0.7)
plt.xlabel('Month')
plt.ylabel('Cost ($)')
plt.title('Monthly Budget Breakdown')
plt.legend()
plt.show()

# CPO Analysis for each Month
plt.figure(figsize=(12, 6))
plt.plot(budget_df['Month'], budget_df['CPO'], marker='o', color='purple', label='CPO')
plt.axhline(y=50000 / total_orders.sum(), color='red', linestyle='--', label='Target CPO (based on $50M Budget)')
plt.xlabel('Month')
plt.ylabel('Cost Per Order ($)')
plt.title('Cost Per Order (CPO) vs Target Budget')
plt.legend()
plt.show()

# Recommendations for Cost Management
# Check if forecasted total cost exceeds the budget
quarterly_total_cost = budget_df['Total Cost'].sum()
target_budget = 50000000  # $50,000,000 for the quarter

if quarterly_total_cost > target_budget:
    print("Warning: Forecasted total cost exceeds target budget.")
    print(f"Forecasted Total Cost: ${quarterly_total_cost:.2f}")
    print(f"Target Budget: ${target_budget:.2f}")
    print("Recommendation: Adjust forecasted demand or explore cost reduction options.")
else:
    print("Forecasted costs are within the target budget.")
