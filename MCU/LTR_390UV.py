import pyb
from machine import I2C, Pin
import time

# I2C address for LTR-390-UV
LTR390_I2C_ADDR = 0x53

# Register addresses (from datasheet)
MAIN_CTRL = 0x00
ALS_UVS_MEAS_RATE = 0x04
ALS_UVS_GAIN = 0x05
PART_ID = 0x06
MAIN_STATUS = 0x07
ALS_DATA = 0x0D
UVS_DATA = 0x10

# Modes
MODE_ALS = 0x00
MODE_UVS = 0x01

INTEGRATION_TIME = 0x20
GAIN_VALUE = 0x03

# I2C setup for Pyboard (Y9=SCL, Y10=SDA)
i2c = I2C(2, freq=400000)


def write_register(reg, value):
    i2c.writeto_mem(LTR390_I2C_ADDR, reg, bytes([value]))

def read_register(reg, length=1):
    return i2c.readfrom_mem(LTR390_I2C_ADDR, reg, length)

def read_20bit_data(reg_base):
    data = read_register(reg_base, 3)
    return data[0] | (data[1] << 8) | (data[2] << 16)

def init_sensor():
    part_id = read_register(PART_ID)[0]
    if part_id != 0xB2:
        raise Exception("LTR-390 not found!")

    write_register(MAIN_CTRL, 0x02)  # Enable sensor
    write_register(ALS_UVS_MEAS_RATE, INTEGRATION_TIME)  # 100ms integration time
    write_register(ALS_UVS_GAIN, GAIN_VALUE)       # Gain x3

def read_als_uv():
    # Read ALS
    write_register(MAIN_CTRL, 0x02 | (MODE_ALS << 3))
    time.sleep_ms(120)
    als = read_20bit_data(ALS_DATA)

    # Read UVS
    write_register(MAIN_CTRL, 0x02 | (MODE_UVS << 3))
    time.sleep_ms(120)
    uvs = read_20bit_data(UVS_DATA)

    return als, uvs

# Main loop
init_sensor()

while True:
    als, uvs = read_als_uv()
    print("ALS:", als, "UVS:", uvs)
    time.sleep(1)