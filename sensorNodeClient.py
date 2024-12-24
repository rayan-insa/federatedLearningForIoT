import random
import datetime
import socket
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

stationsCoords = [(41.133, 29.067), (41.25, 29.033), (40.9, 29.15), (40.977, 28.821), (40.97, 28.82), (40.9, 29.31), (40.899, 29.309), (40.667, 29.283)]
stationDic = {i : station for i, station in enumerate(stationsCoords)}
conditions = ["clear", "cloudy", "rain", "snow"]
DEST_IP = socket.gethostbyname(socket.gethostname())
DEST_PORT = 1245
ENCODER = "utf-8"
BYTESIZE = 1024
sendingNumber = 10   


def preprocess_data(data):
    # Extract features and labels
    X = [[d[3], d[4].timestamp(), d[5]] for d in data]  # temperature, timestamp, condition
    y = [d[6] for d in data]  # isHeatIsland

    # Encode categorical data
    le = LabelEncoder()
    X = [[x[0], x[1], le.fit_transform([x[2]])[0]] for x in X]

    return X, y

def trainModel(data, model):
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy:.4f}')

    return model


class SensorNode():

    def __init__(self, stationId, sendingNumber=10):
        self.sendingNumber = sendingNumber
        self.stationId = stationId
        self.generateSensorsData()
        print("All the data is sent by sensor node", self.stationId)

    def generateSensorsData(self):
        # Generate random data
        dataWithoutY = []

        for i in range(self.sendingNumber):
            longitude = stationDic[self.stationId][0]
            latitude = stationDic[self.stationId][1]
            day = random.randint(1, 30)
            month = random.randint(1, 12)
            year = random.randint(2019, 2024)
            timestamp = datetime.datetime(year, month, day)
            if self.stationId <= (len(stationsCoords)-1)/2 :
                if month <= 2 and month >= 11:
                    temperature = random.randint(15, 23)
                else :
                    temperature = random.randint(25, 35)
            else :
                if month <= 2 and month >= 11:
                    temperature = random.randint(5, 15)
                else :
                    temperature = random.randint(20, 30)
            condition = random.choice(conditions)

            
            
            dataWithoutY.append([self.stationId, latitude, longitude, temperature, timestamp, condition])
        
        return dataWithoutY
            # Appel de function qui genere y en fonctiond e data (X)

            # Appel de function d'entrainement du model avc entrainement

        """with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((DEST_IP, DEST_PORT))
            s.send(str(data).encode(ENCODER))

            if i == self.sendingNumber-1:
                s.send('QUIT'.encode(ENCODER))"""
            
            # Enregistrement dans un fichier pth des wieghts
        

if __name__ == "__main__":

    stationId = int(input(f"Enter the stationID you are (0-{len(stationsCoords)-1}): "))
    SensorNode(stationId, sendingNumber)