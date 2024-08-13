import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

customer_df = pd.read_csv('customer_table.csv')
sales_df = pd.read_csv('sale_table.csv')
productDetails_df = pd.read_csv('product_detail_table.csv')
productGroup_df = pd.read_csv('product_group_table.csv')

print("Sales type: ", sales_df.dtypes)

sales_df['TotalAmount'] = sales_df['TotalAmount']

sales_df['SaleDate'] = pd.to_datetime(sales_df['SaleDate'])

dailyRevenue = sales_df.groupby('SaleDate')['TotalAmount'].sum().reset_index()

dailyRevenue['Day'] = dailyRevenue['SaleDate'].dt.day
dailyRevenue['Month'] = dailyRevenue['SaleDate'].dt.month
dailyRevenue['Year'] = dailyRevenue['SaleDate'].dt.year

features = ['Day', 'Month', 'Year']
target = features

X = dailyRevenue[features]
y = dailyRevenue[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Revenue')
plt.plot(y_pred, label='Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.xlabel('Date Index')
plt.ylabel('Revenue')
plt.legend()
plt.show()

future_dates = pd.date_range(start='2023-08-01', end='2023-12-31', freq='D')
future_df = pd.DataFrame({'Date': future_dates})

future_df['Day'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month
future_df['Year'] = future_df['Date'].dt.year

future_features = future_df[features]
future_df['Revenue_forecast'] = model.predict(future_features)

plt.figure(figsize=(12, 6))
plt.plot(future_df['Date'], future_df['Revenue_forecast'], marker='o')
plt.title('Forecast Future Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue Forecast')
plt.grid()
plt.show()