import json
import serial
import time
from SerialListenerClass import SerialListener
from DataContainerClass import DataContainer
from SerialManagerClass import SerialManager
from AlgorithmManagerClass import AlgorithmManager
import numpy as np


# number of data points to average for each sensor before
# sending the data to the algorithms
numberOfDataPoints = 1
numberOfLoops = 100

# Open the serial connection (replace with the correct port name)
portName = "/dev/cu.usbmodem334E355C31332"
print(f"Port name: {portName}")

# Initialisation
dataContainer = DataContainer()
algorithmManager = AlgorithmManager(dataContainer)
serialManager = SerialManager(dataContainer, maxData=100)
serialManager.setPortName(portName)
# serialManager.setPortName(portName)

# for loop pour simuler le call de la séquence plusieurs fois par l'interface
for i in range(numberOfLoops):
    print(f"Loop {i}")
    start = time.time()

    serialManager.updateDataFromMCU(numberOfDataPoints)
    # print(f"Data received from MCU: {serialManager.dataContainer.wavelengthCounts}")

    # Appel des algorithmes dans la classe AlgorithmManager
    # algorithmManager.calculatePosition()
    algorithmManager.calculateWavelength()
    # algorithmManager.calculatePower()

    # newPosition = dataContainer.position
    newWavelength = dataContainer.wavelength
    # newPower = dataContainer.power

    end = time.time()

    # print(f"Réception des données brutes et calcul des nouvelles valeurs en {end-start} ms")
    # print(f"La nouvelle valeur de position est {newPosition}")
    print(f"La nouvelle valeur de longueur d'onde est {newWavelength}")
    print("\n")
    # print(f"La nouvelle valeur de puissance est {newPower}")
    # print(f"La température du heatsink est {dataContainer.temperature[-1]}")

    time.sleep(0.001)  # 1ms delay to simulate the time between each loop






