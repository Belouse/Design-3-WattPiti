import ujson # normal error here
import sys


class MCUSerialPort():

    def __init__(self):
        pass

    def send(self, formattedData):
        jsonstr = ujson.dumps(formattedData)
        sys.stdout.write(jsonstr + "\n")  # Send data over USB serial (VCP)