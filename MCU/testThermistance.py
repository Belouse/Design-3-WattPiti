import pyb # normal error here
from machine import I2C # normal error here
from startindicator import startIndicator
from MUXClass import Mux
from ThermalMatrixClass import ThermalMatrix


# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting

muxPin1 = "Y7"    # LSB
muxPin2 = "Y8"
muxPin3 = "X11"
muxPin4 = "X12"   # MSB

thermalMatrixPin = "X22"


# ---------- THERMAL MATRIX ----------

mux = Mux(muxPin1, muxPin2, muxPin3, muxPin4)
thermalMatrix = ThermalMatrix(thermalMatrixPin, mux)
delayBetweenReadings = 1000000 # µsec



while True:
    #       ----- THERMAL MATRIX -----
    readings = thermalMatrix.readChannel(thermistanceNumber, delay=delayBetweenReadings)
    print(readings)
