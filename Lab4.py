import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Cargar y preparar el dataset
data = pd.read_csv('monthly_sunspots.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Normalizar los datos utilizando MinMaxScaler para escalar entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
data['Sunspots'] = scaler.fit_transform(data[['Sunspots']])

# Crear ventanas de tiempo para la serie temporal
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 12
X, y = create_dataset(data['Sunspots'].values, time_step)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Separar en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Definir el modelo FFNN
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(time_step, 1)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mae')
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Graficar la pérdida
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.show()

# Predicciones
y_pred = model.predict(X_test)
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# Definir el modelo RNN
model_rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(time_step, 1)),
    keras.layers.SimpleRNN(50, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model_rnn.compile(optimizer='adam', loss='mae')
model_rnn.summary()

# Entrenar el modelo RNN
history_rnn = model_rnn.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Graficar la pérdida
plt.plot(history_rnn.history['loss'], label='Train Loss')
plt.plot(history_rnn.history['val_loss'], label='Test Loss')
plt.legend()
plt.show()

# Predicciones con el modelo RNN
y_pred_rnn = model_rnn.predict(X_test)
plt.plot(y_test, label='Actual')
plt.plot(y_pred_rnn, label='Predicted')
plt.legend()
plt.show()
