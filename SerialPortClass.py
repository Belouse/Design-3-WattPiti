from JSONFormatterClass import JSONFormatter


class SerialPort():

    def __init__(self):
        pass

    def get_data_from_mcu(self):

        # utilisation de la communication série entre le pc et le MCU

        # pour l'instant, on suppose que les données sont ceux là
        # ultiment le microcontrolleur va formatter de la même façon à partir de la classe
        # JSONFormatter et va placer les données des capteurs à l'intérieur avant de l'envoyer
        # par le port série

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

        data = JSONFormatter()
        data.set_data(thermique_Value, λ, i2c_uv, i2c_vis)
        return data.get_data()