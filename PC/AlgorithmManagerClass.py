from Algo.AlgoPosition import AlgoPosition
from Algo.AlgoPuissance import AlgoPower
from Algo.AlgoLambda import AlgoWavelength

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


    def calculatePosition(self):
        """
        Calculate the position of the beam using the data in the DataContainer

        Associate the result of the calculation to the DataContainer.position
        """
        position = self.algoPosition.calculatePosition(self.dataContainer)
        self.dataContainer.position = position
    
    def calculatePower(self):
        """
        Calculate the position of the beam using the data in the DataContainer

        Associate the result of the calculation to the DataContainer.power
        """
        power = self.algoPower.calculatePower(self.dataContainer)
        self.dataContainer.power = power
    
    def calculateWavelength(self):
        """
        Calculate the position of the beam using the data in the DataContainer

        Associate the result of the calculation to the DataContainer.wavelength
        """
        wavelength = self.algoWavelength.calculateWavelength(self.dataContainer)
        self.dataContainer.wavelength = wavelength


