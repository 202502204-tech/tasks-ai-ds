import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
print("loading csv file...")
df = pd.read_csv(url)

print("head of data:")
print(df.head(3))
print(df.info())

null_counts = df.isnull().sum()
print("Nulls: ", null_counts)
df = df.dropna()

X = df[['rm']]
y = df['medv']

print("some basic stats:")
print(df[['rm', 'medv']].describe())

plt.figure(figsize=(7, 4))
sns.scatterplot(x='rm', y='medv', data=df, color='purple', alpha=0.5)
plt.title('Rooms vs Price')
plt.xlabel('Rooms')
plt.ylabel('Price ($1000s)')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Intercept:", lr.intercept_)
print("Coefficient:", lr.coef_)

predictions = lr.predict(X_test)

the_mae = mean_absolute_error(y_test, predictions)
the_mse = mean_squared_error(y_test, predictions)
the_rmse = np.sqrt(the_mse)

print("MAE is:", the_mae)
print("MSE:", the_mse)
print("RMSE:", the_rmse)

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test')

domain = np.linspace(X['rm'].min(), X['rm'].max(), 50).reshape(-1, 1)
range_vals = lr.predict(domain)
plt.plot(domain, range_vals, color='black', linewidth=2, label='Model Line')
plt.xlabel('Average Rooms')
plt.ylabel('Median Value')
plt.legend()
plt.show()