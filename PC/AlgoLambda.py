import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
import os
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import torch

from DataContainerClass import DataContainer
from JupyterLongueurDonde.CapteursDataProcess import DataPreProcess
from JupyterLongueurDonde.NN_Pytorch_Lambda import WavelengthPredictor

class AlgoWavelength:
    """
    Classe pour charger et utiliser un modèle entraîné pour prédire la longueur d'onde
    à partir des valeurs des capteurs.
    """

    def __init__(self, model_path: str = 'model_nn_pytorch_weights.pth'):
        """
        Initialise l'algorithme de prédiction de longueur d'onde en chargeant le modèle préentraîné.

        :param model_path: Chemin vers le fichier du modèle entraîné (str)
        """
        # Obtenir le chemin du répertoire contenant le script en cours d'exécution
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construire le chemin absolu vers le fichier modèle
        model_path_abs = os.path.join(script_dir, model_path)

        print(f"Recherche du modèle à : {model_path_abs}")

        # Vérifier que le fichier du modèle existe
        if not os.path.exists(model_path_abs):
            raise FileNotFoundError(f"Le fichier modèle '{model_path_abs}' n'existe pas")

        # Charger le modèle entraîné
        try:
            self.model = WavelengthPredictor(dropout_rate=0.11276492753388213)  # Recreate the model
            self.model.load_state_dict(torch.load(model_path))
            # Mettre le modèle en mode évaluation
            self.model.eval()
            # print(self.model)
            # with open(model_path_abs, 'rb') as f:
            #    self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")

        # Stocker l'ordre des capteurs pour référence
        self.sensor_order = ['P_IR1', 'P_IR1xP', 'P_IR2', 'P_UV', 'C_UV', 'C_VISG','C_VISB', 'C_VISR']

        # Offset pour mise à zéro
        self.zero_offset = np.zeros(len(self.sensor_order))

        # Initialiser les sensor values
        self.sensor_values = np.zeros(len(self.sensor_order))

        # Initialiser moving window size
        self.moving_window_size = None

        # Créer une instance de EntrainementLambda pour avoir accès aux réponses des capteurs
        self.data_preprocess = DataPreProcess()
        self.angular_dict = self.data_preprocess.angular_dict
        self.responses = self.data_preprocess.all_sensors
        self.response_dict = self.data_preprocess.dict_capteurs

    def angular_factor(self, faisceau_pos=(0, 0, 0)):
        """
        Calcule le facteur géométrique pour chaque capteur en fonction de la position du faisceau.

        :param faisceau_pos: Position du faisceau (x, y, z) pour le calcul de l'angle
        :return: Liste des facteurs géométriques pour chaque capteur (list)
        """
        # Parcourir tous les capteurs et calculer les angles
        geo_factor_list = []

        for sensor_name in self.sensor_order:
            f_x, f_y, f_z = faisceau_pos
            p_x, p_y, p_z = self.response_dict[sensor_name]['position']

            # Calcul de l'angle entre le faisceau et le capteur
            angle = (180 / np.pi) * (np.arccos(abs(p_z - f_z) /
                                               np.sqrt((f_x - p_x) ** 2 + (f_y - p_y) ** 2 + (f_z - p_z) ** 2)))

            distance = np.sqrt((f_x - p_x) ** 2 + (f_y - p_y) ** 2 + (f_z - p_z) ** 2)

            angles = self.angular_dict[sensor_name]['angles']
            intensite_450_interp = self.angular_dict[sensor_name]['intensite_450nm']
            intensite_976_interp = self.angular_dict[sensor_name]['intensite_976nm']
            intensite_1976_interp = self.angular_dict[sensor_name]['intensite_1976nm']

            # Trouver les intensités relatives correspondantes pour chaque longueur d'onde
            # Pour 450nm
            idx_450 = np.abs(angles - angle).argmin()
            intensite_450 = intensite_450_interp[idx_450]

            # Pour 976nm
            idx_976 = np.abs(angles - angle).argmin()
            intensite_976 = intensite_976_interp[idx_976]

            # Pour 1976nm
            idx_1976 = np.abs(angles - angle).argmin()
            intensite_1976 = intensite_1976_interp[idx_1976]

            ref_factor_450, ref_factor_976, ref_factor_1976 = self.response_dict[sensor_name]['geo_factor']

            # Calculer le facteur géométrique
            geo_factor = []
            for i in range(3):
                if i == 0:
                    geo_factor.append(ref_factor_450 / (intensite_450 / (distance ** 2)))
                elif i == 1:
                    geo_factor.append(ref_factor_976 / (intensite_976 / (distance ** 2)))
                elif i == 2:
                    geo_factor.append(ref_factor_1976 / (intensite_1976 / (distance ** 2)))

            geo_factor_list.append(geo_factor)

        return geo_factor_list

    def calculateWavelength(self, sensor_values, faisceau_pos=(0, 0, 0),
                             correction_factor_ind=0,
                             moving_window_size: int = None, enable_print=False):
        """
        Prédit la longueur d'onde à partir des valeurs des capteurs.

        :param sensor_values: Liste ou tableau des valeurs normalisées des capteurs
        dans l'ordre [P_IR1, P_IR1xP, P_IR2, P_UV, C_UV, C_VISG, C_VISB, C_VISR]
        :param faisceau_pos: Position du faisceau (x, y, z) pour le calcul de l'angle
        :param correction_factor_ind: Indice du facteur de correction à appliquer
        :param moving_window_size: Taille de la fenêtre mobile pour le moyennage (int)
        :param enable_print: Si True, affiche les résultats

        :return: Longueur d'onde prédite en nanomètres (float)
        """

        if not moving_window_size:
            self.moving_window_size = sensor_values.shape[0]
        else:
            if moving_window_size > sensor_values.shape[0]:
                self.moving_window_size = sensor_values.shape[0]
            else:
                self.moving_window_size = moving_window_size

        # Calculer la moyenne des n premières valeurs des données reçues
        if sensor_values.shape[0] > 1:
            sensor_values = self.calculate_average_window(sensor_values, self.moving_window_size)

        # Corriger pour le offset de la mise à zéro
        # Assurer que les valeurs ne deviennent pas négatives
        self.sensor_values = np.maximum(0, sensor_values - self.zero_offset)

        # Normaliser les ratios des sensor values
        self._normalize_sensor_values()

        geo_factor_list = self.angular_factor(faisceau_pos)

        for i, k in enumerate(geo_factor_list):
            self.sensor_values_norm[i] = self.sensor_values_norm[i] / k[correction_factor_ind]

        # Convertir le tableau numpy en tensor PyTorch
        tensor_input = torch.tensor(self.sensor_values_norm, dtype=torch.float32)

        # Faire la prédiction
        with torch.no_grad():
            predicted_wavelength = self.model(tensor_input).item()

        if enable_print:
            print("\nTest avec les ratios fournis:")
            print(f"Longueur d'onde prédite: {predicted_wavelength:.2f} nm\n")

        return predicted_wavelength

    def _normalize_sensor_values(self):
        """
        Normalise les valeurs des capteurs en divisant par la valeur maximale.
        """
        self.sensor_values_norm = self.sensor_values / np.max(self.sensor_values)

    def mise_a_zero(self):
        """
        Met à zéro les valeurs des capteurs → Soustraire le background
        pour mesure de longueur d'onde
        """
        self.zero_offset = self.sensor_values

    def reset_mise_a_zero(self):
        """
        Réinitialise les valeurs de mise à zéro des capteurs. Offset = 0
        """
        self.zero_offset = np.zeros(len(self.sensor_order))

    @staticmethod
    def calculate_average_window(sensor_values, moving_window_size):
        """
        Calcule la moyenne des premières valeurs d'un tableau 2D sur l'axe 0.

        :param sensor_values: Tableau de valeurs de dimension (n, 8)
        :param moving_window_size: Nombre de lignes à moyenner
        :return: Vecteur de dimension (8,) contenant les moyennes pour chaque capteur
        """
        # Assurer que moving_window_size ne dépasse pas le nombre de lignes disponibles
        window_size = min(moving_window_size, sensor_values.shape[0])

        # Calculer la moyenne sur les window_size premières lignes
        averaged_values = np.mean(sensor_values[:window_size], axis=0)

        return averaged_values


# exemple de dataset pour tester ici
if __name__ == "__main__":
    # mettre dequoi de cohérent comme valeur ici
    data = DataContainer(wavelengthCounts=np.array([1,2,3,4,5,6,7,8]))

    algo = AlgoWavelength()
    wavelength = algo.calculateWavelength(data.wavelengthCounts,
                                           faisceau_pos=(0, 0, 0),
                                           correction_factor_ind=0,
                                           moving_window_size=3,
                                           enable_print=True)

    print(wavelength)