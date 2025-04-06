import numpy as np
import serial.tools.list_ports
from SerialListenerClass import SerialListener


class SerialManager():
    def __init__(self, dataContainer, maxData=100):
        self.dataContainer = dataContainer
        self.maxData = maxData
        self.serialListener = None


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


    def updateProcessedDataToContainer(self, temperatureArray, wavelengthCountsArray):
        self.dataContainer.temperature = temperatureArray
        self.dataContainer.wavelengthCounts = wavelengthCountsArray


    def convertCountsToTemperature(self, thermalData):
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

        # will work if thermalData is a vector or a matrix
        return thermalData * gains


    def updateDataFromMCU(self, numberOfData, printExecutionTime=True):
        if self.serialListener is None:
            raise Exception("Serial port not set. Please set the port name first.")
        
        rawData = self.serialListener.readData(numberOfData, printExecutionTime)
        print(rawData)
        thermalCountsMatrix, wavelengthCountsMatrix = self.formatData(rawData)
        temperatureMatrix = self.convertCountsToTemperature(thermalCountsMatrix)


        thermalCountsArray, wavelengthArray = self.processData(thermalCountsMatrix, wavelengthCountsMatrix)
        temperatureArray = self.convertCountsToTemperature(thermalCountsArray)

        self.updateRawDataToDataContainer(temperatureMatrix, wavelengthCountsMatrix)
        self.updateProcessedDataToContainer(temperatureArray, wavelengthArray)


    def updateRawDataToDataContainer(self, temperatureMatrix, wavelengthCountsMatrix):
        oldRawTemperatureMatrix = self.dataContainer.rawTemperatureMatrix
        oldRawWavelengthMatrix = self.dataContainer.rawWavelengthMatrix


        # thermal data
        if oldRawTemperatureMatrix.size > 1:
            # if the thermal matrix is not empty, add new data to old ones
            self.dataContainer.rawTemperatureMatrix = np.vstack([temperatureMatrix, oldRawTemperatureMatrix])
        else:
            # assign the new one to the DataContainer field
            self.dataContainer.rawTemperatureMatrix = temperatureMatrix

        # wavelength data 
        if oldRawWavelengthMatrix.size > 1:
            # if the wavelength matrix is not empty, add new data to old ones
            self.dataContainer.rawWavelengthMatrix = np.vstack([wavelengthCountsMatrix, oldRawWavelengthMatrix])
        else:
            # assign the new one to the DataContainer field
            self.dataContainer.rawWavelengthMatrix = wavelengthCountsMatrix

        # remove rows if there are more than self.maxData
        if self.dataContainer.rawTemperatureMatrix.shape[0] > self.maxData:
            self.dataContainer.rawTemperatureMatrix = self.dataContainer.rawTemperatureMatrix[:self.maxData]

        if self.dataContainer.rawWavelengthMatrix.shape[0] > self.maxData:
            self.dataContainer.rawWavelengthMatrix = self.dataContainer.rawWavelengthMatrix[:self.maxData]


    def setPortName(self, portName):
        if self.serialListener is None:
            self.serialListener = SerialListener(portName)
            print("Pas de listener")
        else:
            print("On update le listener")
            self.serialListener.updatePortName(portName)


    def closePort(self):
        self.serialListener.closePort()