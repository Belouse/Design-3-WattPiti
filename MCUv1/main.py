import pyb # normal error here
import startindicator
from PhotodiodeClass import PhotoDiode
from SerialPortClass import MCUSerialPort
from JSONFormatterClass import JSONFormatter
from MUXClass import Mux


# Start up of the MCU
print("Pyboard start up...")
startindicator.startIndicator()

# Initialization of the sensors
photoDiode1 = PhotoDiode("X19")
photoDiode2 = PhotoDiode("X20")
photoDiode3 = PhotoDiode("X21")
photoDiode4 = PhotoDiode("X22")

# Initialization of the MUX
mux = Mux("X9", "X10", "X11", "X12")

# Initialization of the serial port for communication with a computer
serialPort = MCUSerialPort()
jsonFormatter = JSONFormatter()


while True:
    readingPhotoDiode1 = photoDiode1.read()
    readingPhotoDiode2 = photoDiode2.read()
    readingPhotoDiode3 = photoDiode3.read()
    readingPhotoDiode4 = photoDiode4.read()
    photoDiodeReadings = [readingPhotoDiode1,readingPhotoDiode2, readingPhotoDiode3, readingPhotoDiode4]

    i2cReadings = []
    thermalReadings = []


    formattedData = jsonFormatter.format_data(thermalReadings, photoDiodeReadings, i2cReadings)

    serialPort.send(formattedData)

