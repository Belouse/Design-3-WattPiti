import pyb # normal error here
from machine import I2C # normal error here
from startindicator import startIndicator
from PhotodiodeClass import PhotoDiode
from MCUSerialPortClass import MCUSerialPort
from JSONFormatterClass import JSONFormatter
from MUXClass import Mux
from ThermalMatrixClass import ThermalMatrix
from MCP9808Class import MCP9808
from LTR390Class import LTR_390
# from VELM6040 import VEML6040


# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting

# ------- PINS -------
photoDiode1Pin = "X19"  # MTPD2601T-100
photoDiode2Pin = "X20"  # MTPD3001D3-030 sans verre
photoDiode3Pin = "X21"  # MTPD3001D3-030 avec verre
photoDiode4Pin = "X22"  # 019-101-411

muxPin1 = "X7"    # LSB
muxPin2 = "X8"
muxPin3 = "X9"
muxPin4 = "X10"   # MSB

thermalMatrixPin = "X12"


# Initialization of the sensors

# ------- PHOTODIODES -------
photoDiode1 = PhotoDiode(photoDiode1Pin) # MTPD2601T-100
photoDiode2 = PhotoDiode(photoDiode2Pin) # MTPD3001D3-030 sans verre
photoDiode3 = PhotoDiode(photoDiode3Pin) # MTPD3001D3-030 avec verre
photoDiode4 = PhotoDiode(photoDiode4Pin) # 019-101-411

# ------- THERMAL MATRIX -------
mux = Mux(muxPin1, muxPin2, muxPin3, muxPin4)
thermalMatrix = ThermalMatrix(thermalMatrixPin, mux)
delayBetweenReadings = 100 # µsec

# ------- I2C SENSORS -------

# ---- I2C bus ----
i2c = I2C(2, freq=400000)

# ---- LTR_390 ----
#  Choices for resolution: 
    #           13 bits (integration time = 12.5ms),
    #           16 bits (inegration time = 25ms), 
    #           17 bits (integration time = 50ms), 
    #           18 bits (integration time = 100ms), 
    #           19 bits (integration time = 200ms),
    #           20 bits (integration time = 400ms)
ltr390_resolution = 20

# Choices for gain: 1, 3, 6, 9 or 18
ltr390_gain = 18

ltr390 = LTR_390(i2c, ltr390_resolution, ltr390_gain)

mcp9808 = MCP9808(i2c)
# velm6040 = VEML6040(i2c)

# ------- SERIAL PORT -------
serialPort = MCUSerialPort()
jsonFormatter = JSONFormatter()


while True:
    # Photodiodes analog readings
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

    # VEML6040
    vemlRed = 1000
    vemlGreen = 1001
    vemlBlue = 1002
    vemlWhite = 1003

    # LTR_390
    ltr390_als_reading, ltr390_uv_reading = ltr390.get_als_and_uv_readings()

    # List of wavelength readings
    wavelengthReadings = [readingPhotoDiode1,
                            readingPhotoDiode2, 
                            readingPhotoDiode3,
                            vemlRed,
                            vemlGreen,
                            vemlBlue,
                            vemlWhite,
                            readingPhotoDiode4,
                            ltr390_uv_reading,
                            ltr390_als_reading]
    
    # Thermal matrix readings
    thermalReadings = thermalMatrix.readMatrix(delay=delayBetweenReadings)

    # Heatsink temperature reading
    mcp9808Temp = mcp9808.readTemperature()

    # Thermal readings list
    thermalReadings.append(mcp9808Temp)

    # Format and send data to PC
    formattedData = jsonFormatter.format_data(thermalReadings, wavelengthReadings)
    serialPort.send(formattedData)
    pyb.delay(100)