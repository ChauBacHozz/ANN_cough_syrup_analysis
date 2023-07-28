# Import thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ANN_pkg_2
from sklearn.preprocessing import scale 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
df = pd.read_excel('mtnd.xlsx').values
X, y = df[:, 4:],  df[:, 1:4] / 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_scaled, X_test_scaled = scale(X_train), scale(X_test)
X_train_scaled, X_test_scaled = X_train_scaled.T,  X_test_scaled.T
y_train, y_test = y_train.T, y_test.T
model = ANN_pkg_2.Neural_Network([X_train_scaled.shape[0],10, 10,y_train.shape[0]], ANN_pkg_2.ReLU)
model.fit(X_train_scaled, y_train, X_test_scaled, y_test, 
learning_rate = 0.04, alpha = 0, 
epochs = 200000, lr_down=True, lr_decay=100)
print("Train MSE value", model.cost_his[-1])
print("Test MSE value", model.test_cost_his[-1])
plt.plot(model.cost_his, label = "Train loss")
plt.plot(model.test_cost_his, label = "Validation")
plt.legend(loc="upper right")
plt.show()
