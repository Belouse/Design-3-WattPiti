
from pyb import Pin

class Mux():
    def __init__(self, pin1, pin2, pin3, pin4):
        self.pins = [
            Pin(pin1, Pin.OUT),  # Least significant bit (LSB)
            Pin(pin2, Pin.OUT),
            Pin(pin3, Pin.OUT),
            Pin(pin4, Pin.OUT)   # Most significant bit (MSB)
        ]

    def channelSensor(self, sensorNumber):
        if not (1 <= sensorNumber  <= 16):
            raise Exception("MUX channel number is out of range. Should be between 1 and 16.")

        sensorNumber -= 1  # Map 1-16 to 0-15
        for i in range(4):
            value = (sensorNumber >> i) & 1
            self.pins[i].value(value)  # Extract bit i and set pin
            print(value, end='')

# mux = Mux("X9", "X10", "X11", "X12")
# for i in range(1,17):   
#     mux.channelSensor(i)
#     print("\n")



