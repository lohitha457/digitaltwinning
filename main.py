import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic temperature data for the physical object (sensor)
np.random.seed(0)
time_steps = 100
temperature_actual = 20 + np.cumsum(np.random.randn(time_steps))

# Simulate noise and anomalies in the data
temperature_actual += np.random.normal(scale=0.5, size=time_steps)
temperature_actual[20:25] += 5  # Simulate an anomaly (temperature spike)

# Create a digital twin by copying the actual temperature data
temperature_twin = temperature_actual.copy()

# Perform AI-based predictions on the digital twin
X = np.arange(time_steps).reshape(-1, 1)
y = temperature_twin.reshape(-1, 1)

# Train a linear regression model as the AI component
model = LinearRegression()
model.fit(X, y)
temperature_predicted = model.predict(X).flatten()

# Visualization of the actual and predicted temperatures
plt.figure(figsize=(10, 6))
plt.plot(temperature_actual, label="Actual Temperature")
plt.plot(temperature_predicted, label="Predicted Temperature", linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Sensor Digital Twin with AI Prediction")
plt.legend()
plt.grid(True)
plt.show()
