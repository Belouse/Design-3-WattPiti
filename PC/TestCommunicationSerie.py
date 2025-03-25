import json
import serial
import time

# Open the serial connection (replace with the correct port name)
portName = "/dev/cu.usbmodem3976347232332"
ser = serial.Serial(portName, 115200)  # Update with the correct port

dataRead = []
start = time.time()
while len(dataRead) < 10:
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            data = json.loads(line)  # Parse JSON
            dataRead.append(data)
        except json.JSONDecodeError:
            print("Error decoding JSON:", line)
print(f"Temps d'execution {time.time() - start}")
print(dataRead)