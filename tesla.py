import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

dato = pd.read_csv('Tesla.csv' , index_col='Date', parse_dates=['Date'])

print (dato.head)


training = dato[:'2015'].iloc[:, 1:2]
validate = dato['2016':].iloc[:, 1:2]

training['High'].plot(legend=True)
validate['High'].plot(legend=True)
plt.legend(['training 2010 - 2015', 'Validate 2016 en adelante'])
plt.show()

sc = MinMaxScaler(feature_range=(0,1))
scaled = sc.fit_transform(training)

time_step = 60

X_train = []
Y_train = []

for i in range(time_step, len(scaled)):
    X_train.append(scaled[i - time_step:i, 0])
    Y_train.append(scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

input = (X_train.shape[1], 1)
wayout = 1

neuronas = 50

modelo = Sequential()
modelo.add(LSTM(units=neuronas, input_shape=input))
modelo.add(Dense(units=wayout))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train, Y_train, epochs=25, batch_size=32)

x_test = validate.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step, len(x_test)):
    X_test.append(x_test[i - time_step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predict = modelo.predict(X_test)
predict = sc.inverse_transform(predict)

plt.plot(validate.values[0:len(predict)], color='blue', label='real value')
plt.plot(predict, color='green', label='predict')
plt.ylim(1.1 * np.min(predict)/2, 1.1 * np.max(predict))
plt.xlabel('time')
plt.ylabel('real value')
plt.legend()
plt.show()

