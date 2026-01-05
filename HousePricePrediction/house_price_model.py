import pandas as pd
import tensorflow as tf
import numpy as np

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Create dataset
data = {
    'size': [50, 60, 70, 80, 90],
    'rooms': [1, 2, 2, 3, 3],
    'price': [150, 200, 250, 300, 350]
}
df = pd.DataFrame(data)

# Step 2: Separate features and target
X = df[['size', 'rooms']].to_numpy(dtype=np.float32)  # Convert to NumPy array
y = df['price'].to_numpy(dtype=np.float32)

# Step 3: Create normalization layer
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X)  # Fit to training data (NumPy array)

# Step 4: Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),  # Define input shape
    normalizer,                   # Normalize input
    tf.keras.layers.Dense(1)      # Output layer
])

# Step 5: Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mean_squared_error'
)

# Step 6: Train model
model.fit(X, y, epochs=300, verbose=0)

# Step 7: Predict new houses
new_houses = pd.DataFrame({
    'size': [55, 85],
    'rooms': [1, 3]
}).to_numpy(dtype=np.float32)  # Convert to NumPy array

predictions = model.predict(new_houses, verbose=0)

# Step 8: Print results
for i, price in enumerate(predictions.flatten()):
    print(f"Predicted price for house {i+1}: {price:.2f}k")
