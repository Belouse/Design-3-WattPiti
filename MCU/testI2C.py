import pyb # normal error here
from machine import I2C # normal error here
from startindicator import startIndicator

from LTR390Class import LTR_390
from VELM6040Class import VEML6040


# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting

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
# mcp9808 = MCP9808(i2c)


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


names = ["R", "G", "B", "W", "UV", "ALS"]
while True:
    start = pyb.millis()

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
    I2CReadings = [
                    veml6040RedReading,
                    veml6040GreenReading,
                    veml6040BlueReading,
                    veml6040WhiteReading,
                    ltr390_uv_reading,
                    ltr390_als_reading]

    print("--------------------")
    print(names)
    print(I2CReadings)
    pyb.delay(1000)
    
    