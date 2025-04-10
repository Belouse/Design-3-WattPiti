import pyb
from machine import I2C, Pin
import time



class LTR_390UV:

    def __init__(self, i2c):

        # I2C address for LTR-390-UV
        # DO NOT CHANGE
        self.LTR390_I2C_ADDR = 0x53

        # Register addresses (from datasheet)
        # DO NOT CHANGE
        self.MAIN_CTRL = 0x00
        self.ALS_UVS_MEAS_RATE = 0x04
        self.ALS_UVS_GAIN = 0x05
        self.PART_ID = 0x06
        self.MAIN_STATUS = 0x07
        self.ALS_DATA = 0x0D
        self.UVS_DATA = 0x10

        # Modes to switch from
        # DO NOT CHANGE
        self.MODE_ALS = 0x00
        self.MODE_UVS = 0x01

        # Parameters of measurement
        self.RESOLUTION = 0x04 # number of bits of the data taken
        self.MEASUREMENT_RATE = 0x05 # period of measurement
        self.MEAS_RATE_VALUE = self.RESOLUTION << 4 | self.MEASUREMENT_RATE
        print(f"{self.RESOLUTION:b}")
        print(f"{self.MEASUREMENT_RATE:b}")
        print(f"{self.MEAS_RATE_VALUE:b}")
        self.GAIN_VALUE = 0x01 # gain of the sensor (depending on luminosity)

        self.i2c = i2c

        part_id = self.read_register(self.PART_ID)[0]
        if part_id != 0xB2:
            raise Exception("LTR-390 not found!")


        self.write_register(self.MAIN_CTRL, 0x02)  # Enable sensor
        self.write_register(self.ALS_UVS_MEAS_RATE, self.MEAS_RATE_VALUE)  # 100ms integration time
        self.write_register(self.ALS_UVS_GAIN, self.GAIN_VALUE)       # Gain x3

    def write_register(self, reg, value):
        self.i2c.writeto_mem(self.LTR390_I2C_ADDR, reg, bytes([value]))

    def read_register(self, reg, length=1):
        return self.i2c.readfrom_mem(self.LTR390_I2C_ADDR, reg, length)

    def read_20bit_data(self, reg_base):
        data = self.read_register(reg_base, 3)
        return data[0] | (data[1] << 8) | (data[2] << 16)

    def read_als_uv(self):
        # Read ALS
        self.write_register(self.MAIN_CTRL, 0x02 | (self.MODE_ALS << 3))
        time.sleep_ms(30)
        als = self.read_20bit_data(self.ALS_DATA)

        # Read UVS
        self.write_register(self.MAIN_CTRL, 0x02 | (self.MODE_UVS << 3))
        time.sleep_ms(30)
        uvs = self.read_20bit_data(self.UVS_DATA)

        return als, uvs



# Main loop
if __name__ == "__main__":
    i2c = I2C(2, freq=400000) 
    ltr_390uv = LTR_390UV(i2c)
    while True:
        start = pyb.millis()
        als, uvs = ltr_390uv.read_als_uv()
        print("ALS:", als, "UVS:", uvs)
        print(pyb.millis()- start)