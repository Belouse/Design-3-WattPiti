import pyb


def startIndicator():
    leds = [pyb.LED(i) for i in range(1,5)]
    n = 0
    for i in range(40):
        n = i % 4
        leds[n].toggle()
        pyb.delay(50)