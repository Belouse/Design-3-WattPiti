# Example of code to send data from the MCU to the PC using USB communication. The data is converted to JSON format before sending it. 
# I also include the range of values for each sensor. 
# The data includes thermique values, lambda values, I2C UV sensor values, and I2C visible sensor values. 
# The time of the data is also included in the JSON data.

import time


class JSONFormatter():
    """ 
    Classe centralisant le formattage de la communication série vers le PC. On peut toujours le modifier 
    ici selon les besoins. Le but est que le formattage est définit ici on l'utilise ailleurs
    """
    def __init__(self):
        """
        time: int, millisecondes depuis le début de l'initialisation
        temperature: array numpy 1x17 [temp capteur1, temp capteur2, ..., temp heatsink]
        wavelengthCounts: array numpy 1x10 [MTPD2601T-100,
                                        MTPD3001D3-030 sans verre, 
                                        MTPD3001D3-030 avec verre, 
                                        VEML6040A3OG R,
                                        VEML6040A3OG G,
                                        VEML6040A3OG B,
                                        VEML6040A3OG W,
                                        019-101-411, 
                                        LTR-390-UV-01 UVS,
                                        LTR-390-UV-01 ALS]
        """

        self.data = {
            "time": 0,
            "thermal": [],
            "wavelength": []
        }


    def format_data(self, thermal_readings, wavelength_readings):

        self.data["time"] = time.time()
        self.data["thermal"] = thermal_readings
        self.data["wavelength"] = wavelength_readings

        return self.data


    def get_last_data(self):
        return self.data



    # def format_photodiode_readings(self, list_of_readings):
    #     keys = self.photodiodes_names
    #     values = list_of_readings
    #     dic = dict(zip(keys, values))

    #     return dic

    # def format_i2c_sensors_readings(self, list_of_readings):
    #     keys = self.i2c_sensors_names
    #     values = list_of_readings
    #     dic = dict(zip(keys, values))

    #     return dic


    # def format_thermal_readings(self, list_of_readings):
    #     keys = [i for i in range(1,17)]
    #     values = list_of_readings
    #     dic = dict(zip(keys, values))

        # return dic



# Example data to send
# thermique_Value = { # 16 thermique values on 12bits
#         "T1": 4001, "T2": 4002, "T3": 4003, "T4": 4004, "T5": 4005, "T6": 4006,
#         "T7": 4007, "T8": 4008, "T9": 4009, "T10": 4010, "T11": 4011, "T12": 4012,
#         "T13": 4013, "T14": 4014, "T15": 4015, "T16": 4016
#     }

#     # 4 lambda values on 12bits
# λ = { 
#         "λ_1": 4001,
#         "λ_2": 4002,
#         "λ_3": 4003,
#         "λ_4": 4004
#     }

# i2c_uv = {
#         "value": 1048575,   # 24bits
#         "gain": 4,  # from 0 to 4
#         "resolution": 5,  # from 0 to 5
#         "measurement_rate": 5  # from 0 to 5
#     }

# i2c_vis = {
#         "R": 65535, # 16bits
#         "G": 65534,
#         "B": 65533,
#         "W": 65532,
#         "integration_time": 5  # from 0 to 5
#     }

# # data = JSONFormatter()
# # data.set_data(thermique_Value, λ, i2c_uv, i2c_vis)

# # json_str = data.to_json()

# # print(json_str)

# # Convert to JSON
# # ser = serial.Serial('/dev/ttyUSB0', 115200) # change port 
# # # I don't know if it work for MacOs ?

# # # Send JSON string with a newline 
# # ser.write((json_str + "\n").encode('utf-8'))

# # ser.close()