import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

# Cargar y preparar el dataset
data = pd.read_csv('monthly_sunspots.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Normalizar los datos
data['Sunspots'] = (data['Sunspots'] - data['Sunspots'].mean()) / data['Sunspots'].std()

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
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Graficar la pérdida
import matplotlib.pyplot as plt
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

