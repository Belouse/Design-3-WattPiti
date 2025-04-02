import pyb # normal error here
from machine import I2C, Pin
import time

# MCP9808 default I2C address
MCP9808_ADDR = 0x18
TEMP_REG = 0x05

# Initialize I2C (check your Pyboard's I2C pins)
i2c = I2C(2, freq=400000)  # I2C bus 1, standard frequency


def read_temperature():
    data = i2c.readfrom_mem(MCP9808_ADDR, TEMP_REG, 2)
    
    raw_temp = (data[0] << 8) | data[1]
    
    # Extract temperature and handle sign bit
    temp_celsius = raw_temp & 0x0FFF
    temp_celsius /= 16.0
    if raw_temp & 0x1000:  # Check sign bit
        temp_celsius -= 256
    
    return temp_celsius


