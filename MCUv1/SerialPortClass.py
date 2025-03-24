import json

class MCUSerialPort():

    def __init__(self):
        pass

    def send(self, data):
        json_data = json.dumps(data)
        print(json_data)