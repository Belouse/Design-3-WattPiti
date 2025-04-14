import pyb # normal error here
from startindicator import startIndicator
from PhotodiodeClass import PhotoDiode
# from VELM6040 import VEML6040
# Start up of the MCU
print("Pyboard start up...")
startIndicator() # lights will blink on the MCU to show that the code excution is starting

# pins
photoDiode1Pin = "X19"  # MTPD2601T-100
photoDiode2Pin = "X20"  # MTPD3001D3-030 sans verre
photoDiode3Pin = "X21"  # MTPD3001D3-030 avec verre
photoDiode4Pin = "X22"  # 019-101-411


# Initialization of the sensors
photoDiode1 = PhotoDiode(photoDiode1Pin)
photoDiode2 = PhotoDiode(photoDiode2Pin)
photoDiode3 = PhotoDiode(photoDiode3Pin)
photoDiode4 = PhotoDiode(photoDiode4Pin)

names = ["IR2_#1", "IR1_#2", "IR1xP_#3", "UV_#4"]
while True:

    # photodiode analog readings
    readingPhotoDiode1 = photoDiode1.read()
    readingPhotoDiode2 = photoDiode2.read()
    readingPhotoDiode3 = photoDiode3.read()
    readingPhotoDiode4 = photoDiode4.read()

    photoReadings = [readingPhotoDiode1,
                            readingPhotoDiode2, 
                            readingPhotoDiode3,
                            readingPhotoDiode4
                            ]
    print("--------------------")
    print(names)
    print(photoReadings)
    pyb.delay(1000)
