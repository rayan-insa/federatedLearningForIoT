import pandas as pd
import numpy as np

# Load your weather dataset
df = pd.read_csv("data/istanbul_weather_data.csv")

# Define sensor locations and adjustments
sensor_locations = {
    "Bosphorus": {"temp_offset": 0.5, "humidity_offset": 5},
    "North_Istanbul": {"temp_offset": -1, "humidity_offset": 5},
    "City_Center": {"temp_offset": 2, "humidity_offset": -5},
}


# Function to simulate localized data
def simulate_sensor_data(sensor_location, data):
    adjusted_data = data.copy()

    # Apply temperature and humidity adjustments based on location
    adjusted_data["MaxTemp"] += sensor_location["temp_offset"]
    adjusted_data["AvgHumidity"] += sensor_location["humidity_offset"]

    # Optionally, we could add more adjustments (e.g., wind, pressure) in the same way
    return adjusted_data


# Create a dataset for each sensor
sensor_data = {}
for location, adjustments in sensor_locations.items():
    # Simulate data for each sensor by applying local adjustments
    sensor_data[location] = simulate_sensor_data(adjustments, df)

    # Save the sensor data for later use
    sensor_data[location].to_csv(f"data/{location}_sensor_data.csv", index=False)


# Example: Viewing simulated data for the City Center
print(sensor_data["City_Center"].head())

# Now, sensor_data dictionary contains the simulated data for each location
