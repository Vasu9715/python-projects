import tensorflow as tf
import numpy as np

# Make results reproducible (important for tests)
np.random.seed(42)
tf.random.set_seed(42)

def create_and_train_model():
    # Training data
    x_train = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    y_train = np.array([1, 3, 5, 7, 9, 11], dtype=np.float32)

    # Build model (modern Keras style)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units=1)
    ])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="mean_squared_error"
    )

    # Train
    model.fit(x_train, y_train, epochs=500, verbose=0)

    return model

def predict_values(model, x_values):
    x_values = np.array(x_values, dtype=np.float32)
    return model.predict(x_values, verbose=0)
