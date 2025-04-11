from machine import I2C
import pyb


class VEML6040:
    def __init__(self, i2c, integration_time):
        self.i2c = i2c

        self.VELM6040_I2C_ADDR = 0x10
        self.R_DATA = 0x08
        self.G_DATA = 0x09
        self.B_DATA = 0x0A
        self.W_DATA = 0x0B
        self.CONF = 0x00

        if integration_time not in [40, 80, 160, 320, 640, 1280]:
            raise Exception("Integration time sould be of 40, 80, 160, 320, 640 or 1280 ms")

        self.integration_time_dic = {
                                    40: 0x00,
                                    80: 0x01,
                                    160: 0x02,
                                    320: 0x03,
                                    640: 0x04,
                                    1280: 0x05
                                    }
        self.INTEGRATION_TIME = self.integration_time_dic[integration_time]
        self.configure()

        self.red_reading = 0
        self.green_reading = 0
        self.blue_reading = 0
        self.white_reading = 0

    def write_register(self, reg, value):
        self.i2c.writeto_mem(self.VELM6040_I2C_ADDR, reg, bytes([value]))


    def read_register(self, reg, length=1):
        return self.i2c.readfrom_mem(self.VELM6040_I2C_ADDR, reg, length)

    
    def read_16bit_data(self, reg_base):
        data = self.read_register(reg_base, 2)
        return data[0] | (data[1] << 8)

    def configure(self):
        # Typical configuration: no trigger, auto mode, 160ms integration, enable sensor
        config = self.INTEGRATION_TIME << 4
        self.write_register(0x00, config)
        pyb.delay(100)  # Wait for sensor to initialize

    def read_rgbw(self):
        r = self.read_16bit_data(self.R_DATA)
        g = self.read_16bit_data(self.G_DATA)
        b = self.read_16bit_data(self.B_DATA)
        w = self.read_16bit_data(self.W_DATA)
        return r, g, b, w

    def update_data(self):
        r, g, b, w = self.read_rgbw()

        self.red_reading = r
        self.green_reading = g
        self.blue_reading = b
        self.white_reading = w

i2c = I2C(2, freq=400000) # I2C bus 1, standard frequency

# Choices for integration time:
    #           40 ms,
    #           80 ms, 
    #           160 ms, 
    #           320 ms, 
    #           640 ms,
    #           1280 ms
integration_time = 1280

veml6040 = VEML6040(i2c, integration_time)

if __name__ == "__main__":
    while True:
        start = pyb.micros()
        veml6040.update_data()
        red_reading = veml6040.red_reading
        green_reading = veml6040.green_reading
        blue_reading = veml6040.blue_reading
        white_reading = veml6040.white_reading

        print(red_reading, green_reading, blue_reading, white_reading)
        print(pyb.micros()- start)