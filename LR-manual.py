# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
class ManualLR:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y): #trains model on given data

        # initialising variables
        samples, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = (X @ self.weights) + self.bias #calculates prediction
            errors = y_pred - y #calculates error

            self.losses.append(np.mean(errors**2)) #appends MSE to losses

            weight_gradient = (1/samples) * (X.T @ errors) #calculates gradients
            bias_gradient = np.mean(errors)

            self.weights -= self.learning_rate*weight_gradient #updates values
            self.bias -= self.learning_rate*bias_gradient


    def predict(self, X): #calculates and return line with values that are trained
        return (X @ self.weights) + self.bias



def manual_train_test_split(df, test_size, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    data = np.array(df)

    n = len(data)
    k = int(n * test_size)

    testing_indices = np.random.choice(n, size=k, replace=False)

    testing_data = data[testing_indices]
    y_test = testing_data[:, 0]
    X_test = np.delete(testing_data, 0, axis=1)


    training_data = np.delete(data, testing_indices, axis=0)
    y_train = training_data[:, 0]
    X_train = np.delete(training_data, 0, axis=1)

    return X_train, X_test, y_train, y_test



def manual_r2(test_data, predicted_data):

    test_mean = np.mean(test_data)
    data_variance = test_data - test_mean
    SStot = np.sum(data_variance ** 2)

    error_variance = test_data - predicted_data
    SSres = np.sum(error_variance ** 2)

    return 1-(SSres/SStot)


    

# %%
X_train, X_test, y_train, y_test = manual_train_test_split(df, test_size=0.2, random_state=7)

#fix to object data type error
X_train = np.asarray(X_train, dtype=np.float64)
y_train = np.asarray(y_train, dtype=np.float64)

#normalising data to fix overflow error
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std = np.where(std == 0, 1.0, std)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


model = ManualLR()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = manual_r2(y_test, y_pred)
print("r2: ", r2)



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


