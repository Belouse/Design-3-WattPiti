import pyb
from MUXClass import Mux
class ThermalMatrix():

    def __init__(self, pinNumber, mux):
        if type(pinNumber) != str:
            raise TypeError("pinNumber input should a string.")
        if len(pinNumber) != 3:
            raise Exception("pinNumber input sould be of the following format X10")
        if type(mux) != Mux:
            raise TypeError("mux input sould be of Mux type")

        self.pinNumber = pinNumber
        self.adc = pyb.ADC(pyb.Pin(str(self.pinNumber)))

        self.mux = mux


    def readChannel(self, channelNumber, delay):
        self.mux.channelSensor(channelNumber)
        pyb.udelay(delay)
        reading = self.adc.read()

        return reading


    def readMatrix(self, delay):
        readings = []

        for i in range(1,17):
            reading = self.readChannel(i, delay)
            readings.append(reading)

        return readings
