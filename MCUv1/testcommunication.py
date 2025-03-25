import pyb
import sys
import time
import machine
import ujson


i = 0
while True:
    data = {"data": [i,i,3,3], "data2": [3,i*i,3,54,i]}
    jsonstr = ujson.dumps(data)
    sys.stdout.write(jsonstr + "\n")  # Send data over USB serial (VCP)
    pyb.delay(10)

    i+=1

