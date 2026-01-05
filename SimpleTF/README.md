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
