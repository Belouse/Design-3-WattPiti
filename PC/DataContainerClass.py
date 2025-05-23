from dataclasses import dataclass, field
import numpy as np


@dataclass
class DataContainer:
    """
    Contient les informations venant des capteurs

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

    position: tuple (x,y)
    power : float
    wavelength float

    rawTemperatureMatrix: array numpy 2D stockant les données sans moyennage [temperature array1, temperature array2, ...]
    rawWavelengthMatrix: array numpy 2D stockant les données sans moyennage [wavelengthCounts array1, wavelengthCountsarray2, ...]


    Cet objet permet de contenir les données importantes et de les mettre à jour à chaque mise
    à jour depuis le MCU. Les algorithmes iront mettre à jour les nouvelles valeurs de puissance,
    position et longueur d'onde. Un seul objet DataContainer est créé au début du code et celui-ci
    est mise à jour durant l'éxecution de code. Les données sont transférées entre les parties du code
    de cette façon. Toutes les parties du code se réfèrent à cet objet pour le partage d'information.
    Lors de l'éxecution du code, l'interface ira seulement lire les valeurs à afficher du DataContainer
    après avoir appelé les fonctions qui iront mettre à jour les valeurs.

    Pour accéder à chacun des éléments, Ex:     
                    
                    dataContainer = DataContainer() # objet qui serait créé dans le code ou passé en argument
                    arrayTemperature = dataContainer.temperature # pour accéder à l'attribut temperature de l'objet
    """

    # The position x,y of each thermal captor, should not change during execution
    thermalCaptorPosition: np.ndarray = field(default_factory=lambda: np.array([
        [[-10.5, 10.5], [-3.5, 10.5], [3.5, 10.5], [10.5, 10.5]],
        [[-10.5, 3.5],  [-3.5, 3.5],  [3.5, 3.5],  [10.5, 3.5]],
        [[-10.5, -3.5], [-3.5, -3.5], [3.5, -3.5], [10.5, -3.5]],
        [[-10.5, -10.5],[-3.5, -10.5],[3.5, -10.5], [10.5, -10.5]]
        ]))
    
    temperature: np.ndarray  = field(default_factory=lambda: np.array([]))
    interpolatedTemperatureGrid: np.ndarray = field(default_factory=lambda: np.array([]))
    max_temperature: float = 0.0
    old_max_temperature: float = 0.0
    wavelengthCounts: np.ndarray  = field(default_factory=lambda: np.array([]))
    position: tuple = (0,0)
    power : float = 1
    photoPower : float = 1
    wavelength: float = 1000
    Delta_t: float = 0.1 # in seconds, time between each update of max_temperature

    rawTemperatureMatrix: np.ndarray = field(default_factory=lambda: np.zeros([]))
    rawWavelengthMatrix: np.ndarray = field(default_factory=lambda: np.zeros([]))

