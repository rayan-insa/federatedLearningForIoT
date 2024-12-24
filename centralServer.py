import socket
from threading import Thread, Lock
from queue import Queue
import torch
import torch.nn as nn
import numpy as np




# Function to aggregate model weights
def aggregate_weights(models):
    """Aggregate the weights of multiple models (from different sensors)."""
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


def makePrediction(data, model):
    """Make a prediction using the model"""
    
    # Convert the data to a tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Ensure the model is in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(data_tensor)

    return predictions

def vizualisation(predictions):
    # Vizualize the prediction in a graph
    return

if __name__ == "__main__":
    # aggregate all the weights files
    # makePrediction
    # vizualisation

    readWeights()
    aggregateWeights()
    makePrediction()
    vizualisation()
