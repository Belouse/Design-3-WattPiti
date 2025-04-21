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
from VELM6040Class import VEML6040


# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting

# ------- PINS -------
photoDiode1Pin = "X20"  # MTPD2601T-100
photoDiode2Pin = "X21"  # MTPD3001D3-030 sans verre
photoDiode3Pin = "X22"  # MTPD3001D3-030 avec verre
photoDiode4Pin = "X2"  # 019-101-411

muxPin1 = "Y7"    # LSB
muxPin2 = "Y8"
muxPin3 = "X11"
muxPin4 = "X12"   # MSB

thermalMatrixPin = "X19"


# Initialization of the sensors


# ---------- PHOTODIODES ----------

photoDiode1 = PhotoDiode(photoDiode1Pin) # MTPD2601T-100
photoDiode2 = PhotoDiode(photoDiode2Pin) # MTPD3001D3-030 sans verre
photoDiode3 = PhotoDiode(photoDiode3Pin) # MTPD3001D3-030 avec verre
photoDiode4 = PhotoDiode(photoDiode4Pin) # 019-101-411


# ---------- THERMAL MATRIX ----------

mux = Mux(muxPin1, muxPin2, muxPin3, muxPin4)
thermalMatrix = ThermalMatrix(thermalMatrixPin, mux)
delayBetweenReadings = 50 # µsec


# ---------- I2C SENSORS ----------

#       ------- I2C bus -------
i2c = I2C(2, freq=400000)

#       ------- LTR_390 UV -------
#  Choices for resolution: 
    #           13 bits (integration time = 12.5ms),
    #           16 bits (inegration time = 25ms), 
    #           17 bits (integration time = 50ms), 
    #           18 bits (integration time = 100ms), 
    #           19 bits (integration time = 200ms),
    #           20 bits (integration time = 400ms)
ltr390_resolution = 20

# Choices for gain: 
    #           x1, 
    #           x3, 
    #           x6, 
    #           x9,
    #           x18
ltr390_gain = 3

ltr390 = LTR_390(i2c, ltr390_resolution, ltr390_gain)


#       ------- MCP9880 -------
mcp9808 = MCP9808(i2c)


#      ------- VEML6040 -------
# Choices for integration time:
    #           40 ms,
    #           80 ms, 
    #           160 ms, 
    #           320 ms, 
    #           640 ms,
    #           1280 ms
integration_time = 1280

veml6040 = VEML6040(i2c, integration_time)


# ------- SERIAL PORT -------
serialPort = MCUSerialPort()
jsonFormatter = JSONFormatter()


while True:
    start = pyb.millis()
    #   ---------- λ SENSORS ----------

    #       ----- PHOTODIODES -----
    readingPhotoDiode1 = photoDiode1.read()
    readingPhotoDiode2 = photoDiode2.read()
    readingPhotoDiode3 = photoDiode3.read()
    readingPhotoDiode4 = photoDiode4.read()

    #      ----- I2C λ SENSORS -----
    #          --- VEML6040 ---
    veml6040.update_data()
    veml6040RedReading = veml6040.red_reading
    veml6040GreenReading = veml6040.green_reading
    veml6040BlueReading = veml6040.blue_reading
    veml6040WhiteReading = veml6040.white_reading
    #          --- LTR-390 ---
    ltr390_als_reading, ltr390_uv_reading = ltr390.get_als_and_uv_readings()

    # List of wavelength readings
    wavelengthReadings = [readingPhotoDiode1,
                            readingPhotoDiode2, 
                            readingPhotoDiode3,
                            veml6040RedReading,
                            veml6040GreenReading,
                            veml6040BlueReading,
                            veml6040WhiteReading,
                            readingPhotoDiode4,
                            ltr390_uv_reading,
                            ltr390_als_reading]

    # ---------- THERMAL SENSORS ----------

    #       ----- THERMAL MATRIX -----
    thermalReadings = thermalMatrix.readMatrix(delay=delayBetweenReadings)

    #          ----- MCP9808 -----
    mcp9808Temp = mcp9808.readTemperature()

    # List of thermal readings
    thermalReadings.append(mcp9808Temp)


    # ---------- DATA TRANSMISSION ----------
    formattedData = jsonFormatter.format_data(thermalReadings, wavelengthReadings)
    serialPort.send(formattedData)