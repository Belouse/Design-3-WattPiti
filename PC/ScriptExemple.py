import json
import serial
import time
from SerialListenerClass import SerialListener
from DataContainerClass import DataContainer
from SerialManagerClass import SerialManager
from AlgorithmManagerClass import AlgorithmManager
import numpy as np
# number of data points to average for each sensor befor
# sending the data to the algorithms
numberOfDataPoints = 1
numberOfLoops = 100

# Open the serial connection (replace with the correct port name)
portName = "/dev/cu.usbmodem3976347232332" 

dataContainer = DataContainer()
algorithmManager = AlgorithmManager(dataContainer)
serialManager = SerialManager(dataContainer, maxData=100)
serialManager.setPortName(portName)
serialManager.setPortName(portName)

# for loop pour simuler le call de la séquence plusieurs fois par l'interface
for i in range(numberOfLoops):
    start = time.time()
    serialManager.updateDataFromMCU(numberOfDataPoints)
    # algorithmManager.calculatePosition()
    # algorithmManager.calculateWavelength()
    # algorithmManager.calculatePower()

    # newPosition = dataContainer.position
    # newWavelength = dataContainer.wavelength
    # newPower = dataContainer.power

    data = dataContainer.wavelengthCounts
    print(data)


    end = time.time()

    # print(f"Réception des données brutes et calcul des nouvelles valeurs en {end-start} ms")
    # print(f"La nouvelle valeur de position est {newPosition}")
    # print(f"La nouvelle valeur de longueur d'onde est {newWavelength}")
    # print(f"La nouvelle valeur de puissance est {newPower}")
    # print(f"La température du heatsink est {dataContainer.temperature[-1]}")






