from simple_tf_model import create_and_train_model, predict_values
import numpy as np

def test_predictions():
    model = create_and_train_model()
    x_test = [6, 7, 8]
    y_pred = predict_values(model, x_test)

    # Convert predictions to a 1D array for easier comparison
    y_pred = y_pred.flatten()

    # Expected values
    expected = np.array([13, 15, 17])

    # Check if predictions are close (tolerance = 0.1)
    np.testing.assert_allclose(y_pred, expected, rtol=0.01, atol=0.1)