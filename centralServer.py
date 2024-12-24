import socket
from threading import Thread, Lock
from queue import Queue
import torch
import torch.nn as nn
import numpy as np

# Define constants to be used
HOST_IP = socket.gethostbyname(socket.gethostname())
HOST_PORT = 1245
ENCODER = "utf-8"
BYTESIZE = 1024

class ClientHandled(Thread):
    def __init__(self, client_socket, client_address, queue):
        super().__init__()
        self.client_socket = client_socket
        self.client_address = client_address
        self.queue = queue

    def run(self):
        print("running client thread")
        print(f"New connection from {self.client_address}")

        while True:
            try:
                message = self.client_socket.recv(BYTESIZE).decode(ENCODER)
                if message:
                    if message == "QUIT":
                        print(f"\nEnding the chat with {self.client_address}...goodbye!")
                        break
                    self.queue.put((self.client_socket, message))
                else:
                    break
            except:
                break

        self.client_socket.close()
        print(f"Connection with {self.client_address} closed")


def listenForData():

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, HOST_PORT))
    server_socket.listen(5)
    server_socket.settimeout(1.0)
    print(f"Server started at {HOST_IP}:{HOST_PORT}")

    queue = Queue()

    while True:
        try:
            client_socket, client_address = server_socket.accept()
            print(f"client {client_address} accepté")
            client_thread = ClientHandled(client_socket, client_address, queue)
            client_thread.start()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"{e}")
            break

    
    server_socket.close()
    print("Server has been stopped.")


def classificationModelTraining():
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = Model()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy data for training
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def modelPrediction():
    pass


if __name__ == "__main__":
    Thread(target=listenForData).start()
