from PC.Algo.AlgoPosition import AlgoPosition
from PC.Algo.AlgoPuissance import AlgoPower
from PC.Algo.AlgoLambda import AlgoWavelength


class AlgorithmManager():
    
    def __init__(self, algoWavelength, algoPower, algoPosition):
        self.algoWavelength = algoWavelength
        self.algoPower = algoPower
        self.algoPosition = algoPosition
        self.data = None


    def set_data(self, data):
        self.data = data


    def calculate_position(self):

    # interpolation de la position à partir des données self.data
        position = self.algoPosition.calculatePosition(self.data)
        return position
    
    def calculate_power(self):
    
    # calcul de la puissance à partir des données self.data
        power = self.algoPower.calculatePower(self.data)
        return power
    
    def calculate_wavelength(self):

    # calcul de la longueur d'onde à partir des données self.data
        wavelength = self.algoWavelength.calculateWavelength(self.data)
        return wavelength


