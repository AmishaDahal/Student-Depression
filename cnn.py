import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

data = pd.read_csv('features.csv')
X = data.drop(columns=['Depression'])  
y = data['Depression'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

X_cnn = X_scaled.reshape(X_scaled.shape[0], 3, 4, 1)

X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=(3, 4, 1)),  
    MaxPooling2D(pool_size=(1, 1)), 
    Dropout(0.25),
    Conv2D(64, (2, 2), activation='relu', padding='same'),  
    MaxPooling2D(pool_size=(1, 1)), 
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, callbacks=[early_stopping, reduce_lr])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
model.save("student_depression_model.h5")
