import numpy as np
from SerialListenerClass import SerialListener


class SerialManager():
    def __init__(self, portName, dataContainer):
        self.serialListener = SerialListener(portName)
        self.dataContainer = dataContainer


    def formatData(self, serialData):
        thermaldata = []
        wavelengthdata = []
        for dic in serialData:
            thermalList = dic["thermal"]
            thermaldata.append(thermalList)

            wavelengthList = dic["wavelength"]
            wavelengthdata.append(wavelengthList)
        
        thermalMatrix = np.array(thermaldata)
        wavelengthCountsMatrix = np.array(wavelengthdata)

        return thermalMatrix, wavelengthCountsMatrix


    def processData(self, thermalMatrix, wavelengthCountsMatrix):
        thermalArray = np.mean(thermalMatrix, axis=0)
        wavelengthCountsArray = np.mean(wavelengthCountsMatrix, axis=0)

        return thermalArray, wavelengthCountsArray


    def updateDataToContainer(self, temperatureArray, wavelengthCountsArray):
        self.dataContainer.temperature = temperatureArray
        self.dataContainer.wavelengthCounts = wavelengthCountsArray

    def convertCountsToTemperature(self, thermalArray):
        # conversion gain from counts to ˚C [˚C/counts]
        gains = np.array([0.01, #sensor 1
                        0.01,   #sensor 2
                        0.01,   #sensor 3
                        0.01,   #sensor 4
                        0.01,   #sensor 5
                        0.01,   #sensor 6
                        0.01,   #sensor 7
                        0.01,   #sensor 8
                        0.01,   #sensor 9
                        0.01,   #sensor 10
                        0.01,   #sensor 11
                        0.01,   #sensor 12
                        0.01,   #sensor 13
                        0.01,   #sensor 14
                        0.01,   #sensor 15
                        0.01,   #sensor 16
                        1       # heatsink temperature sensor (already in ˚C)
                        ])

        return thermalArray * gains


    def updateDataFromMCU(self, numberOfData, printExecutionTime=True):
        rawData = self.serialListener.readData(numberOfData, printExecutionTime)
        thermalCountsMatrix, wavelengthCountsMatrix = self.formatData(rawData)
        thermalCountsArray, wavelengthArray = self.processData(thermalCountsMatrix, wavelengthCountsMatrix)
        temperatureArray = self.convertCountsToTemperature(thermalCountsArray)
        self.updateDataToContainer(temperatureArray, wavelengthArray)



