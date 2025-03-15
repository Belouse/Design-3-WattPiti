# Example of code to send data from the MCU to the PC using USB communication. The data is converted to JSON format before sending it. 
# I also include the range of values for each sensor. 
# The data includes thermique values, lambda values, I2C UV sensor values, and I2C visible sensor values. 
# The time of the data is also included in the JSON data.

import json
import time
import serial
from enum import Enum

class Data(Enum):
    time          = "time"
    thermique     = "thermique"
    wavelength    = "wavelength"
    i2c_uv        = "I2C_UV"
    i2c_vis       = "I2C_Vis"


class JSONFormatter():

# classe centralisant le formattage de la communication série
# on peut toujours le modifier ici selon les besoins
# le but est que le formattage est définit ici on l'utilise ailleurs

    def __init__(self):
        self.data = {
            Data.time.value: 0,
            Data.thermique.value: {},
            Data.wavelength.value: {},
            Data.i2c_uv.value: {},
            Data.i2c_vis.value: {}
        }
        
    def set_data(self, thermique_Value, wavelength_Value, i2c_uv, i2c_vis):
        self.data = {
            Data.time.value: time.time(),
            Data.thermique.value: thermique_Value,
            Data.wavelength.value: wavelength_Value,
            Data.i2c_uv.value: i2c_uv,
            Data.i2c_vis.value: i2c_vis
        }

    def to_json(self):
        return json.dumps(self.data)

    def get_data(self):
        return self.data


# Example data to send
thermique_Value = { # 16 thermique values on 12bits
        "T1": 4001, "T2": 4002, "T3": 4003, "T4": 4004, "T5": 4005, "T6": 4006,
        "T7": 4007, "T8": 4008, "T9": 4009, "T10": 4010, "T11": 4011, "T12": 4012,
        "T13": 4013, "T14": 4014, "T15": 4015, "T16": 4016
    }

    # 4 lambda values on 12bits
λ = { 
        "λ_1": 4001,
        "λ_2": 4002,
        "λ_3": 4003,
        "λ_4": 4004
    }

i2c_uv = {
        "value": 1048575,   # 24bits
        "gain": 4,  # from 0 to 4
        "resolution": 5,  # from 0 to 5
        "measurement_rate": 5  # from 0 to 5
    }

i2c_vis = {
        "R": 65535, # 16bits
        "G": 65534,
        "B": 65533,
        "W": 65532,
        "integration_time": 5  # from 0 to 5
    }

# data = JSONFormatter()
# data.set_data(thermique_Value, λ, i2c_uv, i2c_vis)

# json_str = data.to_json()

# print(json_str)

# Convert to JSON
# ser = serial.Serial('/dev/ttyUSB0', 115200) # change port 
# # I don't know if it work for MacOs ?

# # Send JSON string with a newline 
# ser.write((json_str + "\n").encode('utf-8'))

# ser.close()