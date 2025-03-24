import pyb # normal error here
from startindicator import startIndicator
from PhotodiodeClass import PhotoDiode
from SerialPortClass import MCUSerialPort
from JSONFormatterClass import JSONFormatter
from MUXClass import Mux
from ThermalMatrixClass import ThermalMatrix

# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting

# pins
photoDiode1Pin = "X19"
photoDiode2Pin = "X20"
photoDiode3Pin = "X21"
photoDiode4Pin = "X22"

muxPin1 = "X7"
muxPin2 = "X8"
muxPin3 = "X9"
muxPin4 = "X10"

thermalMatrixPin = "X12"

# Initialization of the sensors
photoDiode1 = PhotoDiode(photoDiode1Pin)
photoDiode2 = PhotoDiode(photoDiode2Pin)
photoDiode3 = PhotoDiode(photoDiode3Pin)
photoDiode4 = PhotoDiode(photoDiode4Pin)

# Initialization of the thermal matrix
mux = Mux(muxPin1, muxPin2, muxPin3, muxPin4)
thermalMatrix = ThermalMatrix(thermalMatrixPin, mux)
delayBetweenReadings = 100 # µsec

# Initialization of the serial port for communication with a computer
serialPort = MCUSerialPort()
jsonFormatter = JSONFormatter()


while True:
    start = pyb.millis()
    readingPhotoDiode1 = photoDiode1.read()
    readingPhotoDiode2 = photoDiode2.read()
    readingPhotoDiode3 = photoDiode3.read()
    readingPhotoDiode4 = photoDiode4.read()
    photoDiodeReadings = [readingPhotoDiode1,readingPhotoDiode2, readingPhotoDiode3, readingPhotoDiode4]

    thermalReadings = thermalMatrix.readMatrix(delay=delayBetweenReadings)
    i2cReadings = [1000,1000,1000]

    formattedData = jsonFormatter.format_data(thermalReadings, photoDiodeReadings, i2cReadings)

    serialPort.send(formattedData)
    print(f"Loop time: {pyb.millis() - start}  ms")
