import pandas as pd
import tensorflow as tf
import numpy as np

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Dataset
data = {
    'size': [50, 60, 70, 80, 90],
    'rooms': [1, 2, 2, 3, 3],
    'price': [150, 200, 250, 300, 350]
}

df = pd.DataFrame(data)

# Features and target
X = df[['size', 'rooms']].astype(np.float32)
y = df['price'].astype(np.float32)

# 🔑 Normalize features
X = (X - X.mean()) / X.std()

# Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(1)
])

# Compile with safer learning rate
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mean_squared_error'
)

# Train
model.fit(X, y, epochs=300, verbose=0)

# New data
new_houses = pd.DataFrame({
    'size': [55, 85],
    'rooms': [1, 3]
}).astype(np.float32)

# Normalize using training stats
new_houses = (new_houses - X.mean()) / X.std()

# Predict
predictions = model.predict(new_houses, verbose=0)

for i, price in enumerate(predictions.flatten()):
    print(f"Predicted price for house {i+1}: {price:.2f}k")
