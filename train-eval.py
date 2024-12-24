from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from model import WeatherModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load sensor data (example: City Center data)
locations = ["Bosphorus", "North_Istanbul", "City_Center"]
files = [f"data/{location}_sensor_data.csv" for location in locations]
sensor_data = [pd.read_csv(file) for file in files]
i = 0
# Train a model for each sensor location
for data in sensor_data:

    # Preprocess data: Normalize features and split into training/testing sets
    scaler = StandardScaler()
    X = data[["MaxTemp", "AvgHumidity", "AvgWind"]]  # Features
    y = data["MaxTemp"]  # Target

    current_location = locations[i]

    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Initialize the model, loss function, and optimizer
    model = WeatherModel()
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5000
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"{current_location}: Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}"
            )

    # Save the trained model
    torch.save(
        model.state_dict(), f"model_weights/{current_location}_weather_model.pth"
    )
    i += 1

    # Set the model to evaluation mode
    model.eval()

    # Test the model
    with torch.no_grad():
        predictions = model(X_test_tensor)

    # Convert predictions from tensor to numpy for easier plotting
    predictions = predictions.numpy()

    # Plotting the predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(
        y_test_tensor.numpy(),
        label="Actual MaxTemp",
        color="blue",
        linestyle="--",
        marker="o",
    )
    plt.plot(
        predictions, label="Predicted MaxTemp", color="red", linestyle="-", marker="x"
    )
    plt.xlabel("Data Points (Test Set)")
    plt.ylabel("Max Temperature (°C)")
    plt.title(f"Actual vs Predicted Max Temperature for {current_location}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to aggregate model weights
def aggregate_weights(models):
    """Aggregate the weights of multiple models (e.g., from different sensors)."""
    # Start with the weights of the first model
    aggregated_weights = models[0].state_dict()

    # Iterate over the models and average their weights
    for key in aggregated_weights:
        # Sum up all weights for each parameter
        weight_sum = torch.zeros_like(aggregated_weights[key])
        for model in models:
            weight_sum += model.state_dict()[key]

        # Average the weights
        aggregated_weights[key] = weight_sum / len(models)

    return aggregated_weights


# Example of federated learning simulation
# Load models from different sensors
model_bosphorus = WeatherModel()
model_bosphorus.load_state_dict(torch.load("model_weights/weather_model_bosphorus.pth"))

model_city_center = WeatherModel()
model_city_center.load_state_dict(
    torch.load("model_weights/weather_model_city_center.pth")
)

model_north_istanbul = WeatherModel()
model_north_istanbul.load_state_dict(
    torch.load("model_weights/weather_model_north_istanbul.pth")
)

# Aggregate the weights of the models
models = [model_bosphorus, model_city_center, model_north_istanbul]
aggregated_weights = aggregate_weights(models)

# Create a new global model and set its weights to the aggregated weights
global_model = WeatherModel()
global_model.load_state_dict(aggregated_weights)

# Save the global model
torch.save(global_model.state_dict(), "model_weights/global_weather_model_.pth")
print(
    "Federated Learning: Global model has been created and saved as 'weather_model_global.pth'"
)