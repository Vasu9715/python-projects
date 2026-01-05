# House Price Prediction with TensorFlow

A simple beginner-friendly example showing how to build, train, and use a TensorFlow (Keras) model to predict house prices based on size and number of rooms.

This project demonstrates:
- Using pandas to create a dataset
- Preparing data for machine learning
- Training a simple linear regression model with TensorFlow
- Preventing `NaN` predictions using feature normalization
- Making predictions on new data

---

## Project Overview

The model learns how house prices depend on:
- **House size** (in square meters)
- **Number of rooms**

It predicts the **house price (in thousands)**.

---

## Example Dataset

| Size (m²) | Rooms | Price (k) |
|----------:|------:|----------:|
| 50 | 1 | 150 |
| 60 | 2 | 200 |
| 70 | 2 | 250 |
| 80 | 3 | 300 |
| 90 | 3 | 350 |

---

## Technologies Used

- Python 3
- TensorFlow / Keras
- Pandas
- NumPy

---

## Requirements

Install the required packages:

```bash
pip install tensorflow pandas numpy
```
## Running the Code

Run the script using:


```bash
python house_price_model.py
```