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
from CapteursDataProcess import DataPreProcess
from NN_Pytorch_Lambda import WavelengthPredictor

class AlgoWavelength:
    """
    Classe pour charger et utiliser un modèle entraîné pour prédire la longueur d'onde
    à partir des valeurs des capteurs.
    """

    def __init__(self, model_path: str = 'model_nn_pytorch_weights3.pth'):
        """
        Initialise l'algorithme de prédiction de longueur d'onde en chargeant le modèle préentraîné.

        :param model_path: Chemin vers le fichier du modèle entraîné (str)
        """
        
        self.algo_path = model_path
        
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
            self.model.load_state_dict(torch.load(model_path_abs))
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
        self.mean_sensor_values = np.zeros_like(self.sensor_values)

        # Initialiser moving window size
        self.moving_window_size = None

        # Créer une instance de EntrainementLambda pour avoir accès aux réponses des capteurs
        self.data_preprocess = DataPreProcess()
        self.angular_dict = self.data_preprocess.angular_dict
        self.responses = self.data_preprocess.all_sensors
        self.response_dict = self.data_preprocess.dict_capteurs
        self.callibration_gains = np.array(self.data_preprocess.callibration['gains'])

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

    def _prepare_sensor_data(self, raw_data):
        """
        Extrait et réorganise les données des capteurs selon l'ordre défini dans self.sensor_order.

        :param raw_data: Données brutes des 10 capteurs (array de taille 10 ou matrice Nx10)
        :return: Données organisées des 8 capteurs dans l'ordre défini par self.sensor_order
        """
        # Mapping entre l'indice dans raw_data et le nom du capteur
        raw_to_sensor_map = {
            1: 'P_IR1',  # MTPD3001D3-030 sans verre
            2: 'P_IR1xP',  # MTPD3001D3-030 avec verre
            0: 'P_IR2',  # MTPD2601T-100
            7: 'P_UV',  # 019-101-411
            8: 'C_UV',  # LTR-390-UV-01 UVS
            4: 'C_VISG',  # VEML6040A3OG G
            5: 'C_VISB',  # VEML6040A3OG B
            3: 'C_VISR'  # VEML6040A3OG R
        }

        # Indices des capteurs à extraire dans l'ordre de self.sensor_order
        indices_to_extract = [idx for idx, sensor in raw_to_sensor_map.items()
                              if sensor in self.sensor_order]

        # Réorganiser les indices selon l'ordre dans self.sensor_order
        ordered_indices = []
        for sensor in self.sensor_order:
            for idx, name in raw_to_sensor_map.items():
                if name == sensor:
                    ordered_indices.append(idx)
                    break

        # Si raw_data est une matrice Nx10
        if len(raw_data.shape) > 1:
            return raw_data[:, ordered_indices]
        # Si raw_data est un vecteur de longueur 10
        else:
            return raw_data[ordered_indices]


    def calculateWavelength(self, dataContainer, faisceau_pos=(0, 0, 0),
                             correction_factor_ind=0,
                             moving_window_size: int = None, enable_print=False):
        """
        Prédit la longueur d'onde à partir des valeurs des capteurs.

        :param dataContainer: DataContainer object (see class declaration for details)
        dans l'ordre [P_IR1, P_IR1xP, P_IR2, P_UV, C_UV, C_VISG, C_VISB, C_VISR]
        :param faisceau_pos: Position du faisceau (x, y, z) pour le calcul de l'angle
        :param correction_factor_ind: Indice du facteur de correction à appliquer
        :param moving_window_size: Taille de la fenêtre mobile pour le moyennage (int)
        :param enable_print: Si True, affiche les résultats

        :return: Longueur d'onde prédite en nanomètres (float)
        """
        self.sensor_values = dataContainer.rawWavelengthMatrix

        # Vérifier si les données brutes contiennent 10 capteurs
        if self.sensor_values.shape[-1] == 10:
            # Extraire et réorganiser les 8 capteurs nécessaires
            self.sensor_values = self._prepare_sensor_data(self.sensor_values)

        # print(f"Longueur sensor values calculate wavelength :{sensor_values}")

        # Moyenner les valeurs de la matrice sur les n lignes de la matrice (dataContainer.rawWavelengthMatrix = sensor_values)
        self.mean_sensor_values = np.mean(self.sensor_values, axis=0)

        # if len(sensor_values.shape) > 1:
        #     if not moving_window_size:
        #         self.moving_window_size = sensor_values.shape[0]
        #     else:
        #         if moving_window_size > sensor_values.shape[0]:
        #             self.moving_window_size = sensor_values.shape[0]
        #         else:
        #             self.moving_window_size = moving_window_size
        #
        #     # Calculer la moyenne des n premières valeurs des données reçues
        #     if sensor_values.shape[0] > 1:
        #         sensor_values = self.calculate_average_window(sensor_values, self.moving_window_size)

        # Corriger pour le offset de la mise à zéro
        # Assurer que les valeurs ne deviennent pas négatives
        self.mean_sensor_values = np.maximum(0, self.mean_sensor_values - self.zero_offset)

        self.sensor_values_with_gain = self.mean_sensor_values * self.callibration_gains

        # Normaliser les ratios des sensor values
        self._normalize_sensor_values()

        geo_factor_list = self.angular_factor(faisceau_pos)

        for i, k in enumerate(geo_factor_list):
            self.sensor_values_norm[i] = self.sensor_values_norm[i] * k[correction_factor_ind]

    
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
        self.sensor_values_norm = self.sensor_values_with_gain / np.max(self.sensor_values_with_gain)
        # self.sensor_values_norm = self.mean_sensor_values / np.max(self.mean_sensor_values)

    def mise_a_zero(self):
        """
        Met à zéro les valeurs des capteurs → Soustraire le background
        pour mesure de longueur d'onde
        """
        self.zero_offset = self.mean_sensor_values

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


=======
    def plot_spectral_ratios(self):
        """
        Affiche les courbes de réponse spectrale des capteurs.
        """
        plt.figure(figsize=(10, 6))
        
        for sensor_name in self.sensor_order:
            sensor_data = self.responses[sensor_name]
            plt.plot(sensor_data[:, 0], sensor_data[:, 1], label=sensor_name)
        
        plt.title("Réponses spectrales des capteurs")
        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Réponse [counts/W]")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_spectral_response(self):
        """
        Affiche les courbes de réponse spectrale des capteurs.
        """
        plt.figure(figsize=(10, 6))
        
        for sensor_name in self.sensor_order:
            sensor_data = self.response_dict[sensor_name]['data']
            plt.plot(sensor_data[:, 0], sensor_data[:, 1], label=sensor_name)
        
        plt.title("Réponses spectrales des capteurs")
        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Réponse normalisée")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def get_sensor_response_for_wavelength(self, wavelength, enable_print=False):
        """
        Extrait la réponse de chaque capteur pour une longueur d'onde donnée.
        
        Parameters:
        wavelength (float): Longueur d'onde en nanomètres pour laquelle extraire les réponses
        
        Returns:
        dict: Dictionnaire contenant les réponses de chaque capteur
        """
        # Dictionnaire pour stocker les réponses
        responses_dict = {}
        
        # Liste pour stocker les réponses dans l'ordre spécifié
        responses_list = []
        
        # Pour chaque capteur, trouver la valeur à la longueur d'onde spécifiée dans self.response_dict
        for sensor_name in self.sensor_order:
            sensor_data = self.response_dict[sensor_name]['data']
            
            # Trouver l'indice le plus proche de la longueur d'onde demandée
            idx = np.abs(sensor_data[:, 0] - wavelength).argmin()
            
            # Obtenir la valeur du capteur à cette longueur d'onde
            response = sensor_data[idx, 1]
            
            # Stocker dans le dictionnaire
            responses_dict[sensor_name] = response
            responses_list.append(response)
            
        if enable_print:
            print("\nRéponses des capteurs pour {} nm:".format(wavelength))
            for sensor, reponse in responses_dict.items():
                print(f"{sensor}: {reponse:.4f}")
    
        return responses_dict, responses_list


# # exemple de dataset pour tester ici
# if __name__ == "__main__": 
#     # Capteurs: ["R","G","B","UV","IR2_#1","IR1_#2","IR1xP_#3","UV_#4"]
#     mis_a_zero_1976 = [10,10,3,0,3.75,3.33,5.25,10.75]
#     mesure_1976_5W_20mm = [8.89,9,3,0,4.56,1627.44,1802.56,12.67] # centre
#     mesure_1976_5W_19mm = [9.69,9.54,2.69,0,4,1594.15,1798.31,18.62]
#     mesure_1976_5W_18mm = [8.17,8,1.5,0,5.58,1575.75,1817,13.33]
    
#     mis_a_zero_976 = [8,8,0,0,3.63,4.38,6.25,11]
#     mesure_976_2_5W = [0,0,0,0,999,153.5,183.13,624]
#     mesure_976_5W = [0,0,0,0.67,2031.56,306.56,351.67,1250.44]
#     mesure_976_7_5W = [0,0,0,2,3093.44,456,523.56,1912.78]
#     mesure_976_10W = [0,0,0,2,4094.54,607.77,690.77,2551]
    
#     mis_a_zero_450 = [11,10.54,3,0,3.23,3.92,5.31,13]
#     mesure_450_2_5W = [3819,5065.63,18840.56,0,13.63,21.44,36.19,224.19]
#     mesure_450_5W = [8543.88,11574.25,34747,0,34.13,45.31,70.56,495.81]
#     mesure_450_7_5W = [11224.42,14078.17,42298.75,0,48.42,61.33,88.42,681.5]
#     mesure_450_10W = [14716.17,19816.5,49570.75,0,66.08,81.83,112.83,907.58]
    
    
#     def re_order(array):
#         # Capteurs: ["R","G","B","UV","IR2_#1","IR1_#2","IR1xP_#3","UV_#4"]
#         # vers =>
#         # ['P_IR1', 'P_IR1xP', 'P_IR2', 'P_UV', 'C_UV', 'C_VISG', 'C_VISB', 'C_VISR']
#         return np.array([array[5], array[6], array[4], array[7], array[3], array[1], array[2], array[0]])
    
#     # [MTPD2601T-100,
#     # MTPD3001D3-030 sans verre, 
#     # MTPD3001D3-030 avec verre, 
#     # VEML6040A3OG R,
#     # VEML6040A3OG G,
#     # VEML6040A3OG B,
#     # VEML6040A3OG W,
#     # 019-101-411, 
#     # LTR-390-UV-01 UVS,
#     # LTR-390-UV-01 ALS]
    
#     def mis_a_zero(array, mis_a_zero):
#         soustrai = np.array(array) - np.array(mis_a_zero)
#         # remplace les valeurs négatives par 0
#         soustrai[soustrai < 0] = 0
#         return soustrai
    
    
    
    
#     mesure_450_2_5W = re_order(mis_a_zero(mesure_450_2_5W, mis_a_zero_450))
#     mesure_450_5W = re_order(mis_a_zero(mesure_450_5W, mis_a_zero_450))
#     mesure_450_7_5W = re_order(mis_a_zero(mesure_450_7_5W, mis_a_zero_450))
#     mesure_450_10W = re_order(mis_a_zero(mesure_450_10W, mis_a_zero_450))
    
#     mesure_976_2_5W = re_order(mis_a_zero(mesure_976_2_5W, mis_a_zero_976))
#     mesure_976_5W = re_order(mis_a_zero(mesure_976_5W, mis_a_zero_976))
#     mesure_976_7_5W = re_order(mis_a_zero(mesure_976_7_5W, mis_a_zero_976))
#     mesure_976_10W = re_order(mis_a_zero(mesure_976_10W, mis_a_zero_976))
    
#     mesure_1976_5W_20mm = re_order(mis_a_zero(mesure_1976_5W_20mm, mis_a_zero_1976))
#     mesure_1976_5W_19mm = re_order(mis_a_zero(mesure_1976_5W_19mm, mis_a_zero_1976))
#     mesure_1976_5W_18mm = re_order(mis_a_zero(mesure_1976_5W_18mm, mis_a_zero_1976))


#     donnees = [mesure_450_2_5W, mesure_450_5W, mesure_450_7_5W, mesure_450_10W,
#               mesure_976_2_5W, mesure_976_5W, mesure_976_7_5W, mesure_976_10W,
#               mesure_1976_5W_20mm, mesure_1976_5W_19mm, mesure_1976_5W_18mm]
    
    
#     def plot_reponses_et_ratios():
#         algo = AlgoWavelength()
    
#         # Afficher les courbes de réponse spectrale
#         algo.plot_spectral_response()
#         algo.plot_spectral_ratios()
        
        
#     predicted_wavelengths = []
#     for i in donnees:
#         data = DataContainer(rawWavelengthMatrix=np.array([i]))
    
#         algo = AlgoWavelength()
#         wavelength = algo.calculateWavelength(data,
#                                                faisceau_pos=(0, 0, 0),
#                                                correction_factor_ind=0,
#                                                moving_window_size=3,
#                                                enable_print=False)
#         predicted_wavelengths.append(wavelength)
    
    
#     predictions_450 = np.array(predicted_wavelengths[0:4])
#     predictions_976 = np.array(predicted_wavelengths[4:8])
#     predictions_1976 = np.array(predicted_wavelengths[8:11])
    
    
    
#     print(predictions_450, predictions_976, predictions_1976)
    
#     plt.figure(figsize=(8,4))
#     plt.xlabel("Puissance [W]")	
#     plt.ylabel("Erreur relative [%]")
    
#     puissance = [2.5, 5, 7.5, 10]
    
#     wavelength = 450
#     plt.plot(puissance,abs(predictions_450-wavelength)/wavelength*100, 'o-', label='450nm')
#     wavelength = 976
#     plt.plot(puissance,abs(predictions_976-wavelength)/wavelength*100, 'o-', label='976nm')
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     plt.figure(figsize=(8,4))
#     plt.xlabel("Position du faisceau par rapport au centre [mm]")
#     plt.ylabel("Erreur relative [%]")
    
#     position = [0, 1, 2]

#     wavelength = 1976
#     plt.plot(position,abs(predictions_1976-wavelength)/wavelength*100, 'o-', label='1976nm')
#     plt.grid(True)
#     plt.legend()
#     plt.show()
    
    # data = DataContainer(rawWavelengthMatrix=np.array([mesure_1976_5W_18mm]))

    # algo = AlgoWavelength()
    # wavelength = algo.calculateWavelength(data,
    #                                        faisceau_pos=(0, 0, 0),
    #                                        correction_factor_ind=0,
    #                                        moving_window_size=3,
    #                                        enable_print=True)

        
        
        
    # plot_reponses_et_ratios()
