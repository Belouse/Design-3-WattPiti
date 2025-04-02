

class MCP9808:

    def __init__(self, i2c):
        self.i2c = i2c
        self.MCP9808_ADDR = 0x18
        self.TEMP_REG = 0x05


    def readTemperature(self):
        data = self.i2c.readfrom_mem(self.MCP9808_ADDR, self.TEMP_REG, 2)
    
        raw_temp = (data[0] << 8) | data[1]
    
        # Extract temperature and handle sign bit
        temp_celsius = raw_temp & 0x0FFF
        temp_celsius /= 16.0
        if raw_temp & 0x1000:  # Check sign bit
            temp_celsius -= 256
    
        return temp_celsius