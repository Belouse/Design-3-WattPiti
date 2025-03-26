import pyb

class PhotoDiode():

    def __init__(self, pinNumber) -> None:
        """
        Constructor of the PhotoDiode class.

        pinNumber : pin number of the ADC for the voltage reading

        """
        if type(pinNumber) != str:
            raise TypeError("pinNumber argument should a string.")
        if len(pinNumber) != 3:
            raise Exception("pinNumber sould be of the following format X10")
        
        self.pinNumber = pinNumber
        self.adc = pyb.ADC(pyb.Pin(str(self.pinNumber)))

    def read(self):
        """
        Read data from the appropriate pin and return it.
        """
        reading = self.adc.read()
        # reading = 10

        return reading