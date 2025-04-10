import pyb # normal error here
from machine import I2C # normal error here
from startindicator import startIndicator
from PhotodiodeClass import PhotoDiode
from MCUSerialPortClass import MCUSerialPort
from JSONFormatterClass import JSONFormatter
from MUXClass import Mux
from ThermalMatrixClass import ThermalMatrix
from MCP9808Class import MCP9808
# from VELM6040 import VEML6040
# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting


muxPin1 = "Y7"
muxPin2 = "Y8"
muxPin3 = "X11"
muxPin4 = "X12"

readPin1 = "X1"
readPin2 = "X2"
readPin3 = "X3"
readPin4 = "X4"

pin1 = pyb.ADC(readPin1)
pin2 = pyb.ADC(readPin2)
pin3 = pyb.ADC(readPin3)
pin4 = pyb.ADC(readPin4)


# Initialization of the thermal matrix
mux = Mux(muxPin1, muxPin2, muxPin3, muxPin4)

while True:
    for i in range(1,17):
        mux.channelSensor(i)
        pyb.delay(1)
        reading1 = pin1.read()
        reading2 = pin2.read()
        reading3 = pin3.read()
        reading4 = pin4.read()
        print(reading4, reading3, reading2, reading1)
        pyb.delay(1000)
