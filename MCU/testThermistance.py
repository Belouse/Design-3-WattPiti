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

thermalMatrixPin = "X19"

thermistanceNumber = 2 # from 1 to 16

# ---------- THERMAL MATRIX ----------S

mux = Mux(muxPin1, muxPin2, muxPin3, muxPin4)
thermalMatrix = ThermalMatrix(thermalMatrixPin, mux)
delayBetweenReadings = 100000 # µsec



while True:
    #       ----- THERMAL MATRIX -----
    reading = thermalMatrix.readChannel(thermistanceNumber, delay=delayBetweenReadings)
    print(f"Reading #{thermistanceNumber}: {reading}")
