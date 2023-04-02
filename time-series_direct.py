import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def create_recursive_data(data, feature, window_size, target_size):
    i = 1
    while i < window_size:
        data["{}_{}".format(feature, i)] = data[feature].shift(-i)
        i += 1

    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data[feature].shift(-window_size-i)
        i += 1

    data = data.dropna(axis=0)
    return data

data = pd.read_csv('./datasets/co2.csv')
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data.co2.interpolate()

window_size = 5
target_size = 3
data = create_recursive_data(data, "co2", window_size, target_size)
target = ["target_{}".format(i) for i in range(target_size)]
# target = "target"
X = data.drop([target, "time"], axis=1)
y = data[target]
# split data
train_size = 0.8
num_samples = len(X)
X_train = X[:int(train_size * num_samples)]
X_test = X[int(train_size * num_samples):]
y_train = y[:int(train_size * num_samples)]
y_test = y[int(train_size * num_samples):]
#
# reg = LinearRegression()
# reg.fit(X_train, y_train)
# y_predict = reg.predict(X_test)
# print("R2 score: ", r2_score(y_test, y_predict))
# print("MSE: ", mean_squared_error(y_test, y_predict))
# print("MAE: ", mean_absolute_error(y_test, y_predict))
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(data["time"][:int(train_size * num_samples)], data["co2_4"][:int(num_samples * train_size)], label="train")
# ax.plot(data["time"][int(train_size * num_samples):], data["co2_4"][int(train_size * num_samples):],
#         label="test", linewidth=3)
# ax.plot(data["time"][int(train_size * num_samples):], y_predict, label="predict")
# ax.set_xlabel("time")
# ax.set_ylabel("co2")
# ax.legend()
# ax.grid()
# plt.show()
