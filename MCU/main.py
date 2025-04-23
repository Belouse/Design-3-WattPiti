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

# ----- ACQUISITION -----
period = 100
numberOfDataPoints = 40

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
integration_time = 40

veml6040 = VEML6040(i2c, integration_time)


# ------- SERIAL PORT -------
serialPort = MCUSerialPort()
jsonFormatter = JSONFormatter()


while True:
    # ------ TOTAL LAMBDA ------
    totalPhoto1 = 0
    totalPhoto2 = 0
    totalPhoto3 = 0
    totalPhoto4 = 0
    totalvemlR = 0
    totalvemlG = 0
    totalvemlB = 0
    totalvemlW = 0
    totalltrUV = 0
    totalltrALS = 0

    # ------ TOTAL THERMAL ------
    therm1 = 0
    therm2 = 0
    therm3 = 0
    therm4 = 0
    therm5 = 0
    therm6 = 0
    therm7 = 0
    therm8 = 0
    therm9 = 0
    therm10 = 0
    therm11 = 0
    therm12 = 0
    therm13 = 0
    therm14 = 0
    therm15 = 0
    therm16 = 0
    mcptot = 0

    start = pyb.millis()
    for i in range(numberOfDataPoints):
        #   ---------- λ SENSORS ----------

        #       ----- PHOTODIODES -----
        totalPhoto1 += photoDiode1.read()
        totalPhoto2 += photoDiode2.read()
        totalPhoto3 += photoDiode3.read()
        totalPhoto4 += photoDiode4.read()

        #      ----- I2C λ SENSORS -----
        #          --- VEML6040 ---
        veml6040.update_data()
        totalvemlR += veml6040.red_reading
        totalvemlG += veml6040.green_reading
        totalvemlB += veml6040.blue_reading
        totalvemlW += veml6040.white_reading
        #          --- LTR-390 ---
        ltr390_als_reading, ltr390_uv_reading = ltr390.get_als_and_uv_readings()

        totalltrUV += ltr390_uv_reading
        totalltrALS += ltr390_als_reading

        # ---------- THERMAL SENSORS ----------

        #       ----- THERMAL MATRIX -----
        thermalData = thermalMatrix.readMatrix(delay=delayBetweenReadings)

        therm1 += thermalData[0]
        therm2 += thermalData[1]
        therm3 += thermalData[2]
        therm4 += thermalData[3]
        therm5 += thermalData[4]
        therm6 += thermalData[5]
        therm7 += thermalData[6]
        therm8 += thermalData[7]
        therm9 += thermalData[8]
        therm10 += thermalData[9]
        therm11 += thermalData[10]
        therm12 += thermalData[11]
        therm13 += thermalData[12]
        therm14 += thermalData[13]
        therm15 += thermalData[14]
        therm16 += thermalData[15]
        #  ----- MCP9808 -----
        mcptot += mcp9808.readTemperature()

    # List of wavelength readings
    wavelengthTotal = [
                        totalPhoto1,
                        totalPhoto2, 
                        totalPhoto3,
                        totalvemlR,
                        totalvemlG,
                        totalvemlB,
                        totalvemlW,
                        totalPhoto4,
                        totalltrUV,
                        totalltrALS]
    
    # List of thermal total
    thermalTotal = [
                    therm1,
                    therm2,
                    therm3,
                    therm4,
                    therm5,
                    therm6,
                    therm7,
                    therm8,
                    therm9,
                    therm10,
                    therm11,
                    therm12,
                    therm13,
                    therm14,
                    therm15,
                    therm16,
                    mcptot]


    for i, lambdaData in enumerate(wavelengthTotal):
        wavelengthTotal[i] = lambdaData/numberOfDataPoints

    for i, thermalData in enumerate(thermalTotal):
        thermalTotal[i] = thermalData/numberOfDataPoints

    executionTime = pyb.millis() - start
    pyb.delay(period - executionTime)

    # ---------- DATA TRANSMISSION ----------
    formattedData = jsonFormatter.format_data(thermalTotal, wavelengthTotal)
    serialPort.send(formattedData)
    print(f"Period: {pyb.millis() - start}")