from AlgoPosition import AlgoPosition
from AlgoPuissance import AlgoPower
from AlgoLambda import AlgoWavelength
from time import time

class AlgorithmManager():
    """
    Manager permettant de coordonner les calculs de longueur d'onde, de puissance et de position
    """
    def __init__(self, dataContainer):
        """
        Constructeur de la classe

        dataContainer: DataContainer object (see class declaration for details)

        Association d'un objet DataContainer qui contient les données.
        Les algos ont accès aux données des capteurs à partir des objets et mettent
        à jour les valeurs de longueur d'onde, puissance et position directement à même
        l'objet à chaque fois que les nouvelles valeurs sont calculées.
        """
        self.algoWavelength = AlgoWavelength()
        self.algoPower = AlgoPower()
        self.algoPosition = AlgoPosition()
        self.dataContainer = dataContainer
        self.time = time()

    def update_time(self):
        """
        Update the time between each calculation of max_temperature
        """
        
        # update the time with the time since the last update
        self.dataContainer.Delta_t = time() - self.time
        self.time = time()


    def calculatePosition(self):
        """
        Calculate the position of the beam using the data in the DataContainer

        Associate the result of the calculation to the DataContainer.position
        """

        self.update_time()

        position, max_temp = self.algoPosition.calculatePosition(self.dataContainer)
        self.dataContainer.position = position
        self.dataContainer.old_max_temperature = self.dataContainer.max_temperature
        self.dataContainer.max_temperature = max_temp
    
    def calculatePower(self):
        """
        Calculate the position of the beam using the data in the DataContainer

        Associate the result of the calculation to the DataContainer.power
        """
        power = self.algoPower.calculer_puissance(self.dataContainer)
        self.dataContainer.power = power
    
    def calculateWavelength(self):
        """
        Calculate the position of the beam using the data in the DataContainer

        Associate the result of the calculation to the DataContainer.wavelength
        """
        # wavelength = self.algoWavelength.calculateWavelength(self.dataContainer.wavelengthCounts)
        wavelength, photoPower = self.algoWavelength.calculateWavelength(self.dataContainer)

        self.dataContainer.wavelength = wavelength
        self.dataContainer.photoPower = photoPower


