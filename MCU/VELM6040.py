from machine import I2C
import time

class VEML6040:
    def __init__(self, i2c, addr=0x10):
        self.i2c = i2c
        self.addr = addr
        self.configure()

    def write_register(self, reg, value):
        data = bytearray([value & 0xFF, (value >> 8) & 0xFF])
        self.i2c.mem_write(data, self.addr, reg)

    def read_register(self, reg):
        data = self.i2c.mem_read(2, self.addr, reg)
        return data[0] | (data[1] << 8)

    def configure(self):
        # Typical configuration: no trigger, auto mode, 160ms integration, enable sensor
        config = 0x0000  # Default: auto mode, 160ms, enable
        self.write_register(0x00, config)
        time.sleep(0.2)  # Wait for sensor to initialize

    def read_rgbw(self):
        r = self.read_register(0x08)
        g = self.read_register(0x09)
        b = self.read_register(0x0A)
        w = self.read_register(0x0B)
        return {'red': r, 'green': g, 'blue': b, 'white': w}


i2c = I2C(2, freq=4000) # I2C bus 1, standard frequency
sensor = VEML6040(i2c)

if __name__ == "__main__":
    while True:
        colors = sensor.read_rgbw()
        print("R: {}, G: {}, B: {}, W: {}".format(colors['red'], colors['green'], colors['blue'], colors['white']))
        time.sleep(1)