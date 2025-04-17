import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import joblib

# Paso 1: Cargar los datos
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

X = np.array(data_dict['data'])
y = np.array(data_dict['labels'])

# Paso 2: Codificar etiquetas (A, B, C, ...) como números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Guardar el codificador para usarlo luego en Android o backend
joblib.dump(label_encoder, 'label_encoder.pkl')

# Paso 3: Dividir datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded
)

# Paso 4: Crear modelo en TensorFlow
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Guardar el modelo original en formato H5
model.save('lsm_model.h5')

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open('lsm_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modelo entrenado y exportado como 'lsm_model.tflite'")