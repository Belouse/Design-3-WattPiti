from DataContainerClass import DataContainer
import numpy as np


class AlgoWavelength():

    def __init__(self):
        pass
        #importer le réseau de neurones???
    
    def calculateWavelength(self, dataContainer):
        """
        dataContainer: DataContainer object (see class declaration for details)

        Fonction qui retourne seulement une valeur ici, pas de graphiques pour ne pas interrompre le main svp
        """


        # la magie se passe ici...

        wavelength = 1000

        return wavelength


# exemple de dataset pour tester ici
if __name__ == "__main__":
    # mettre dequoi de cohérent comme valeur ici
    data = DataContainer(wavelengthCounts=np.array([1,2,3,4,5,6,7,8,9,10]))

    algo = AlgoWavelength()
    wavelength = algo.calculateWavelength(data)

    print(wavelength)