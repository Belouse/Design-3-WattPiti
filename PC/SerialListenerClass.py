import serial
import json
import time

class SerialListener():

    def __init__(self, portName):
        self.port = serial.Serial(portName, 115200)

    def updatePortName(self, portName):
        self.closePort()
        self.port = serial.Serial(portName, 115200)
    
    def readData(self, numberOfData, printExecutionTime):
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
        if printExecutionTime:
            print(f"Temps d'execution de la réception de données {time.time() - start}")

        return dataRead
    
    def closePort(self):
        self.port.close()