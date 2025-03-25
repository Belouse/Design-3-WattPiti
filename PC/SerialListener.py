import serial
import json
import time

class SerialListener():

    def __init__(self, portName):
        self.port = serial.Serial(portName, 115200)

    def updatePort(self, portName):
        self.port = serial.Serial(portName, 115200)
    
    def readData(self, numberOfData):
        dataRead = []
        start = time.time()
        while len(dataRead) < numberOfData:
            line = self.port.readline().decode('utf-8').strip()
            if line:
                try:
                    data = json.loads(line)  # Parse JSON
                    dataRead.append(data)
                except json.JSONDecodeError:
                    pass
        print(f"Temps d'execution de la réception de données {time.time() - start}")
        return dataRead


portName = "/dev/cu.usbmodem3976347232332"
serialPort = SerialListener(portName)
data = serialPort.readData(30)
print(data)