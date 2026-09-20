# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
df = pd.read_csv('Housing.csv')
df

# %%
binaryCols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binaryCols] = df[binaryCols].replace({'yes': 1, 'no': 0})
# df['furnishingstatus'] = df['furnishingstatus'].replace({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)
df

# %%
y = df['price']
X = df.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'mse: {mse} \n r2: {r2}')



# %%

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Model predictions against actual values')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit')


plt.xlabel('Actual Price')
plt.ylabel('Predited Price')
plt.title('Actual vs Predicted Price')
plt.legend()
plt.show()


