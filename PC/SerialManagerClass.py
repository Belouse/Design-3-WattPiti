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
        # gains = np.array([0.01, #sensor 1
        #                 0.01,   #sensor 2
        #                 0.01,   #sensor 3
        #                 0.01,   #sensor 4
        #                 0.01,   #sensor 5
        #                 0.01,   #sensor 6
        #                 0.01,   #sensor 7
        #                 0.01,   #sensor 8
        #                 0.01,   #sensor 9
        #                 0.01,   #sensor 10
        #                 0.01,   #sensor 11
        #                 0.01,   #sensor 12
        #                 0.01,   #sensor 13
        #                 0.01,   #sensor 14
        #                 0.01,   #sensor 15
        #                 0.01,   #sensor 16
        #                 1       # heatsink temperature sensor (already in ˚C)
        #                 ])

        # Définition des coefficients polynomiaux pour chaque capteur
        polynomial_coeffs = [
            [62.188902, -3.17287820e-02, 1.49928777e-05, -6.27900747e-09, 9.37501583e-13],  # sensor 1
            [62.289902, -3.25939928e-02, 1.59024220e-05, -6.55529322e-09, 9.62456358e-13],  # sensor 2
            [62.389902, -3.28848765e-02, 1.62254374e-05, -6.65099599e-09, 9.70550839e-13],  # sensor 3
            [62.489902, -3.35107825e-02, 1.68350473e-05, -7.02737518e-09, 1.04342307e-12],  # sensor 4
            [62.589902, -4.24138952e-02, 2.77093265e-05, -1.25440217e-08, 1.99534265e-12],  # sensor 5
            [62.689902, -3.36589431e-02, 1.73354531e-05, -7.23727707e-09, 1.07109567e-12],  # sensor 6
            [62.789902, -3.37569001e-02, 1.74227435e-05, -7.24686403e-09, 1.06883636e-12],  # sensor 7
            [62.889902, -3.28350700e-02, 1.63313407e-05, -6.76293383e-09, 9.95280512e-13],  # sensor 8
            [62.989902, -3.28881370e-02, 1.62854026e-05, -6.81344890e-09, 1.01375370e-12],  # sensor 9
            [63.089902, -3.37242966e-02, 1.73387854e-05, -7.22276271e-09, 1.06745299e-12],  # sensor 10
            [63.189902, -3.80667490e-02, 2.22867623e-05, -9.75855160e-09, 1.50818228e-12],  # sensor 11
            [63.2899020, -3.36406662e-02, 1.73615672e-05, -7.31357235e-09, 1.09175488e-12],  # sensor 12
            [63.389902, -3.34505131e-02, 1.70295021e-05, -7.17324673e-09, 1.07232115e-12],  # sensor 13
            [63.489902, -3.35112783e-02, 1.71160855e-05, -7.21705519e-09, 1.07965880e-12],  # sensor 14
            [63.589902, -3.26067129e-02, 1.60898771e-05, -6.77901830e-09, 1.01499758e-12],  # sensor 15
            [63.689902, -3.17073636e-02, 1.50420364e-05, -6.33211566e-09, 9.49447029e-13],  # sensor 16
            [0, 0, 0, 0, 0]  # sensor 17 (heatsink, déjà en °C)
        ]

        temperatureData = np.zeros_like(thermalData, dtype=float)

        # Si thermalData est un vecteur (1D)
        if thermalData.ndim == 1:
            for i in range(len(thermalData)):
                if i < 16:
                    temperatureData[i] = np.polyval(polynomial_coeffs[i][::-1],
                                                    thermalData[i])
                else:
                    temperatureData[i] = thermalData[i]

        # Si thermalData est une matrice (2D)
        elif thermalData.ndim == 2:
            for i in range(thermalData.shape[1]):
                if i < 16:
                    temperatureData[:, i] = np.polyval(polynomial_coeffs[i][::-1],
                                                       thermalData[:, i])
                else:
                    temperatureData[:, i] = thermalData[:, i]

        # will work if thermalData is a vector or a matrix
        # return thermalData * gains
        return temperatureData


    def updateDataFromMCU(self, numberOfData, printExecutionTime=True):
        if self.serialListener is None:
            raise Exception("Serial port not set. Please set the port name first.")
        
        rawData = self.serialListener.readData(numberOfData, printExecutionTime)
        thermalCountsMatrix, wavelengthCountsMatrix = self.formatData(rawData)
        temperatureMatrix = self.convertCountsToTemperature(thermalCountsMatrix)


        thermalCountsArray, wavelengthArray = self.processData(thermalCountsMatrix, wavelengthCountsMatrix)
        temperatureArray = self.convertCountsToTemperature(thermalCountsArray)
        # print(f"temp array: {temperatureArray}")

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
            # print("Pas de listener")
        else:
            # print("On update le listener")
            self.serialListener.updatePortName(portName)


    def closePort(self):
        self.serialListener.closePort()