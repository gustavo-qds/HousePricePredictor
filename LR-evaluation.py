# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone

# %%
df = pd.read_csv('Housing.csv')
binaryCols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binaryCols] = df[binaryCols].replace({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].replace({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True, dtype=int)
df

# %%
def evaluate(models, shuffles, test_sizes):
    results = []

    for base_model in models:
        scores = [[], [], []]
        for shuffle in shuffles:
            for test_size in test_sizes:
                

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=shuffle)

                model = clone(base_model)
                model.fit(X_train, y_train)

                model_pred = model.predict(X_test)

                scores[0].append(r2_score(y_test, model_pred))
                scores[1].append(mean_squared_error(y_test, model_pred))
                scores[2].append(mean_absolute_error(y_test, model_pred))

        results.append([base_model, np.mean(scores[0]), np.mean(scores[1]), np.mean(scores[2])])

    return pd.DataFrame(results, columns=["model", "r2", "mse", "mae" ])


# %%
y = df['price']
X = df.drop('price', axis=1)

models = [LinearRegression(), Ridge(), Lasso(), SGDRegressor()]
shuffles = [54, 23, 5, 987]
test_sizes = [0.1, 0.15, 0.2, 0.25]

x = evaluate(models, shuffles, test_sizes)
print(x)


