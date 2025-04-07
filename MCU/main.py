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

# pins
photoDiode1Pin = "X19"  # MTPD2601T-100
photoDiode2Pin = "X20"  # MTPD3001D3-030 sans verre
photoDiode3Pin = "X21"  # MTPD3001D3-030 avec verre
photoDiode4Pin = "X22"  # 019-101-411

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

# Initialization of the serial port for communication with computer
serialPort = MCUSerialPort()
jsonFormatter = JSONFormatter()

# Initialize I2C and the related sensors
i2c = I2C(2, freq=400000)  # I2C bus 1, standard frequency
mcp9808 = MCP9808(i2c)
# velm6040 = VEML6040(i2c)


while True:
    # start = pyb.millis()

    # photodiode analog readings
    readingPhotoDiode1 = photoDiode1.read()
    readingPhotoDiode2 = photoDiode2.read()
    readingPhotoDiode3 = photoDiode3.read()
    readingPhotoDiode4 = photoDiode4.read()

    # i2c reading
    # colors = velm6040.read_rgbw()
    # vemlRed = colors['red']
    # vemlGreen = colors['green']
    # vemlBlue = colors['blue']
    # vemlWhite = colors['white']

    vemlRed = 1000
    vemlGreen = 1001
    vemlBlue = 1002
    vemlWhite = 1003
    ltr390UVS = 1004
    ltr390ALS = 1005

    wavelengthReadings = [readingPhotoDiode1,
                            readingPhotoDiode2, 
                            readingPhotoDiode3,
                            readingPhotoDiode4,
                            vemlRed,
                            vemlGreen,
                            vemlWhite,
                            readingPhotoDiode4,
                            ltr390UVS,
                            ltr390ALS]
    
    thermalReadings = thermalMatrix.readMatrix(delay=delayBetweenReadings)
    mcp9808Temp = mcp9808.readTemperature()

    thermalReadings.append(mcp9808Temp)
    formattedData = jsonFormatter.format_data(thermalReadings, wavelengthReadings)
    serialPort.send(formattedData)
    # print(f"Loop time: {pyb.millis() - start}  ms")
