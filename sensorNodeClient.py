import random
import datetime
import socket

stationsCoords = [(41.133, 29.067), (41.25, 29.033), (40.9, 29.15), (40.977, 28.821), (40.97, 28.82), (40.9, 29.31), (40.899, 29.309), (40.667, 29.283)]
stationDic = {i : station for i, station in enumerate(stationsCoords)}
conditions = ["clear", "cloudy", "rain", "snow"]
DEST_IP = socket.gethostbyname(socket.gethostname())
DEST_PORT = 1245
ENCODER = "utf-8"
BYTESIZE = 1024
sendingNumber = 10   


class SensorNode():

    def __init__(self, stationId, sendingNumber=10):
        self.sendingNumber = sendingNumber
        self.stationId = stationId
        self.generateSensorsData()
        print("All the data is sent by sensor node", self.stationId)

    def generateSensorsData(self):
        # Generate random data
        data = []

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
            data.append([self.stationId, latitude, longitude, temperature, timestamp, condition])


            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((DEST_IP, DEST_PORT))
                s.send(str(data).encode(ENCODER))

                if i == self.sendingNumber-1:
                    s.send('QUIT'.encode(ENCODER))
            
        return
        

if __name__ == "__main__":

    stationId = int(input(f"Enter the stationID you are (0-{len(stationsCoords)-1}): "))
    SensorNode(stationId, sendingNumber)