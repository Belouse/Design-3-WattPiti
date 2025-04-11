import pyb
from machine import I2C, Pin
import time



class LTR_390:

    def __init__(self, i2c, resolution, gain):

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
        if resolution not in [20, 19, 18, 17, 16, 13]:
            raise Exception("Resolution should be of 20, 19, 18 ,17, 16 or 13 bits.")

        if gain not in [1, 3, 6, 9, 18]:
            raise Exception("Gain should be of 1, 3, 6, 9 or 18.")
        
        # Bits to send for all specifications 
        # integration time for number of bits resolution
        self.bits_integration_time_dic = {
                                20: 400,
                                19: 200,
                                18: 100,
                                17: 50,
                                16: 25,
                                13: 12.5
                                }
        # Resolution in number of bits
        self.bits_resolution_dic = {
                                20: 0x00,
                                19: 0x01,
                                18: 0x02,
                                17: 0x03,
                                16: 0x04,
                                13: 0x05
                                }
        # Measurement rate of the chip for bits resolution
        self.measurement_rate_dic = {
                                20: 0x04,
                                19: 0x03,
                                18: 0x02,
                                17: 0x01,
                                16: 0x00,
                                13: 0x00
                                }
        self.RESOLUTION = self.bits_resolution_dic[resolution] # number of bits of the data taken
        self.MEASUREMENT_RATE = self.measurement_rate_dic[resolution]
        self.MEAS_RATE_VALUE = self.RESOLUTION << 4 | self.MEASUREMENT_RATE

        self.gain_dic = {
                        1: 0x00,
                        3: 0x01,
                        6: 0x02,
                        9: 0x03,
                        18: 0x04
        }
        self.GAIN_VALUE = self.gain_dic[gain] # gain of the sensor (depending on luminosity)



        self.i2c = i2c

        part_id = self.read_register(self.PART_ID)[0]
        if part_id != 0xB2:
            raise Exception("LTR-390 not found!")


        self.enable_sensor()
        self.set_measurement_rate_resolution()
        self.set_gain()      

        self.als_current_reading = 0
        self.uv_current_reading = 0


    def write_register(self, reg, value):
        self.i2c.writeto_mem(self.LTR390_I2C_ADDR, reg, bytes([value]))


    def read_register(self, reg, length=1):
        return self.i2c.readfrom_mem(self.LTR390_I2C_ADDR, reg, length)


    def read_20bit_data(self, reg_base):
        data = self.read_register(reg_base, 3)
        return data[0] | (data[1] << 8) | (data[2] << 16)


    def enable_sensor(self):
        self.write_register(self.MAIN_CTRL, 0x02)  # Enable sensor


    def set_measurement_rate_resolution(self):
        self.write_register(self.ALS_UVS_MEAS_RATE, self.MEAS_RATE_VALUE)


    def set_gain(self):
        self.write_register(self.ALS_UVS_GAIN, self.GAIN_VALUE) 


    def update_uv_reading(self):
        uv_reading = self.read_20bit_data(self.UVS_DATA)
        self.uv_current_reading = uv_reading
    

    def update_als_reading(self):
        als_reading = self.read_20bit_data(self.ALS_DATA)
        self.als_current_reading = als_reading
    

    def is_data_ready(self):
        status = self.read_register(self.MAIN_STATUS)[0]

        return (status & 0x08) == 0x08


    def set_uv_mode(self):
        self.write_register(self.MAIN_CTRL, 0x02 | (self.MODE_UVS << 3))


    def set_als_mode(self):
        self.write_register(self.MAIN_CTRL, 0x02 | (self.MODE_ALS << 3))


    def is_in_uv_mode(self):
        ctrl = self.read_register(self.MAIN_CTRL)[0]
        return (ctrl & self.MODE_UVS << 3) == self.MODE_UVS << 3


    def is_in_als_mode(self):
        ctrl = self.read_register(self.MAIN_CTRL)[0]
        return (ctrl &  self.MODE_UVS << 3 ) != self.MODE_UVS << 3


    def get_als_and_uv_readings(self):
        is_data_ready = self.is_data_ready()
        if is_data_ready:
            if self.is_in_uv_mode():
                self.update_uv_reading()
                self.set_als_mode()

            else:
                self.update_als_reading()
                self.set_uv_mode()

        als_reading = self.als_current_reading
        uv_reading = self.uv_current_reading
        
        return als_reading, uv_reading


# Main loop
if __name__ == "__main__":
    #  Choices for resolution: 
    #           13 bits (integration time = 12.5ms),
    #           16 bits (inegration time = 25ms), 
    #           17 bits (integration time = 50ms), 
    #           18 bits (integration time = 100ms), 
    #           19 bits (integration time = 200ms),
    #           20 bits (integration time = 400ms)
    resolution = 20

    # Choices for gain: 1, 3, 6, 9 or 18
    gain = 18


    i2c = I2C(2, freq=400000) 
    
    ltr390 = LTR_390(i2c, resolution, gain)

    # ltr_390.set_uv_mode()
    while True:
        start = pyb.micros()
        als_reading, uv_reading = ltr390.get_als_and_uv_readings()
        print(uv_reading, als_reading)
        print(pyb.micros() - start)