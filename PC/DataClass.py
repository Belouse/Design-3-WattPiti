from dataclasses import dataclass
import numpy as np
@dataclass
class SensorOutput:
    """
    temperature: array numpy 1x17 [temp capteur1, temp capteur2, ..., temp heatsink]
    position: tuple (x,y)
    wavelengthCounts: array numpy 1x6 ["MTPD2601T-100",
                                        "MTPD3001D3-030 sans verre", 
                                        "MTPD3001D3-030 avec verre", 
                                        "VEML6040A3OG",
                                        "019-101-411", 
                                        "LTR-390-UV-01"]
    """
    temperature: np.ndarray
    position: tuple
    wavelengthCounts: np.ndarray