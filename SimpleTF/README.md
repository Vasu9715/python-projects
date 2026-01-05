# Simple TensorFlow Model with Pytest

A minimal example project demonstrating how to build, train, and test a simple TensorFlow (Keras) model using NumPy and pytest.

This repository is designed to be:
- Easy to understand
- Deterministic (tests always give the same result)
- Free of unnecessary warnings
- A good reference for testing ML code

---

## Project Structure
```
.
├── simple_tf_model.py        # Model creation, training, and prediction logic
├── test_simple_tf_model.py   # Pytest unit test for model predictions
├── conftest.py               # (Optional) Pytest warning configuration
├── pytest.ini                # (Optional) Pytest warning filters
└── README.md
```
---

## Model Description

The model learns a simple linear relationship:
y = 2x + 1

---

### Training Data

| x | y |
|---|---|
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

The model is implemented using:
- Keras `Sequential` API
- A single `Dense` layer
- Mean Squared Error (MSE) loss
- Stochastic Gradient Descent (SGD) optimizer

---

## Requirements

- Python 3.10+
- TensorFlow
- NumPy
- pytest

---

## Installation

Install the required dependencies:

```bash
pip install tensorflow numpy pytest
```
## Running the Tests

Run all tests using pytest:
```bash
pytest
```