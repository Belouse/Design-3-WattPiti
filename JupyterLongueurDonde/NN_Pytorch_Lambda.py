import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import constants
import matplotlib.pyplot as plt
from time import perf_counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


start_total_time = perf_counter()


class DataPreProcess:
    """
    Classe pour charger et traiter les données des capteurs
    """
    def __init__(self, gains = None, plastic_name: str='Petri'):
        """
        Initialisation de l'objet des courbes de réponses

        :param gains: Liste des gains des capteurs. (Default = None)
        :param plastic_name: Nom du plastique pour la transmission. (Default = 'Petri')
        """

        # Initialiser les courbes des capteurs
        self.UV1 = None  # Courbe de réponse capteur UV1
        self.UV2 = None  # Courbe de réponse capteur UV2
        self.VIS_green = None  # Courbe de réponse capteur VIS_green
        self.VIS_Blue = None  # Courbe de réponse capteur VIS_Blue
        self.VIS_red = None  # Courbe de réponse capteur VIS_red
        self.IR = None  # Courbe de réponse capteur IR
        self.IR2 = None  # Courbe de réponse capteur IR2
        self.plastic_transmission = None  # Courbe de transmission du plastique (petri)

        self.plastic_t_file = None  # Courbes de transmission de tous les plastiques
        self.plastic_name = plastic_name  # Nom du plastique pour la transmission

        # À corriger !!! Calcul de l'aire de la surface des capteurs I2C pour convertir en Counts/W
        self.P_IR1_area = 2
        self.P_IR1xP_area = 2
        self.P_IR2_area = 2
        self.P_UV_area = 2
        self.C_UV_area = 0.28E-3 ** 2
        self.C_VISG_area = 0.2E-3 ** 2
        self.C_VISB_area = 0.2E-3 ** 2
        self.C_VISR_area = 0.2E-3 ** 2

        # Configurer les gains des capteurs
        if gains is None:
            # Valeurs par défaut si aucun gain n'est fourni
            self.gain_IR1 = 2                       # Gain de la photodiode IR 2500
            self.gain_IR1xP = 2                     # Gain de la photodiode IR 2500 PMMA
            self.gain_IR2 = 2                       # Gain de la photodiode IR 1700
            self.gain_UV = 2                        # Gain de la photodiode UV
            self.gain_C_UV = 100 / 4e9 * 4095       # Gain du capteur I2C UV
            self.gain_C_VISG = 100 / 4e11 * 4095    # Gain du capteur I2C VIS channel Green
            self.gain_C_VISB = 100 / 4e11 * 4095    # Gain du capteur I2C VIS channel Blue
            self.gain_C_VISR = 100 / 4e11 * 4095    # Gain du capteur I2C VIS channel Red
        else:
            # Utiliser les gains fournis
            self.gain_IR1 = gains[0]
            self.gain_IR1xP = gains[1]
            self.gain_IR2 = gains[2]
            self.gain_UV = gains[3]
            self.gain_C_UV = gains[4]
            self.gain_C_VISG = gains[5]
            self.gain_C_VISB = gains[6]
            self.gain_C_VISR = gains[7]

        # Créé un dictionnaire pour les capteurs
        self.dict_capteurs = {
            'P_IR1': {'gain': self.gain_IR1, 'sensor_area': self.P_IR1_area},
            'P_IR1xP': {'gain': self.gain_IR1xP, 'sensor_area': self.P_IR1xP_area},
            'P_IR2': {'gain': self.gain_IR2, 'sensor_area': self.P_IR2_area},
            'P_UV': {'gain': self.gain_UV, 'sensor_area': self.P_UV_area},
            'C_UV': {'gain': self.gain_C_UV, 'sensor_area': self.C_UV_area},
            'C_VISG': {'gain': self.gain_C_VISG, 'sensor_area': self.C_VISG_area},
            'C_VISB': {'gain': self.gain_C_VISB, 'sensor_area': self.C_VISB_area},
            'C_VISR': {'gain': self.gain_C_VISR, 'sensor_area': self.C_VISR_area}}

        # Charger des données
        self._load_data()

        # Traiter les données
        self._process_data()


    def _load_data(self):
        """
        Charger les fichiers CSV
        """
        # Obtenir le chemin du répertoire contenant le script en cours d'exécution
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.UV1 = pd.read_csv(os.path.join(script_dir, "UVS.csv"), header=None)
        self.UV2 = pd.read_csv(os.path.join(script_dir, "UV2.csv"), header=None)
        self.VIS_green = pd.read_csv(os.path.join(script_dir, "Green.csv"), header=None)
        self.VIS_Blue = pd.read_csv(os.path.join(script_dir, "Blue.csv"), header=None)
        self.VIS_red = pd.read_csv(os.path.join(script_dir, "Red.csv"), header=None)
        self.IR = pd.read_csv(os.path.join(script_dir, "IR.csv"), header=None)
        self.IR2 = pd.read_csv(os.path.join(script_dir, "IR2.csv"), header=None)
        self.plastic_t_file = pd.read_csv(os.path.join(script_dir, "TransmissionsPlastiques.csv"))

        # Extraction des données de transmission du plastique de petri
        plastic_index = self.plastic_t_file.columns.get_loc(self.plastic_name)  # Trouver l'index de la colonne correspondant au plastique
        plastic_transmission = self.plastic_t_file.iloc[1:, plastic_index + 1].values.astype(np.float64)[::-1] / 100
        wavelength = self.plastic_t_file.iloc[1:, plastic_index].values.astype(np.float64)[::-1]
        self.plastic_transmission = pd.DataFrame(np.column_stack((wavelength, plastic_transmission)))



    def _interpolate_data(self, new_length=10000):
        """
        Interpoler les données pour avoir une longueur uniforme.

        :param new_length: Nouvelle longueur des données (Default = 10000)
        """
        # Grille de longueurs d'onde
        wavelength = np.linspace(250, 2500, new_length)

        # Liste des attributs à interpoler et leurs noms
        datas = [
            ('UV1', self.UV1),
            ('UV2', self.UV2),
            ('VIS_green', self.VIS_green),
            ('VIS_Blue', self.VIS_Blue),
            ('VIS_red', self.VIS_red),
            ('IR', self.IR),
            ('IR2', self.IR2),
            ('plastic_transmission', self.plastic_transmission)
        ]

        # Interpolation des données
        for dataset_name, data in datas:
            # Trier les données par longueur d'onde
            data = data.sort_values(by=data.columns[0])

            # S'il y a des valeurs négatives, les remplacer par 0.
            data[1] = data[1].apply(lambda x: 0 if x < 0 else x)

            # Récupérer les limites du fichier CSV
            min_x = data.iloc[:, 0].min()
            max_x = data.iloc[:, 0].max()

            new_data = np.zeros((new_length, 2))
            new_data[:, 0] = wavelength

            # Créer un masque pour les longueurs d'onde dans la plage
            mask = (wavelength >= min_x) & (wavelength <= max_x)

            # Créer l'interpolation (cubic pour plus de précision)
            f = interp1d(data[0], data[1], kind='linear', bounds_error=False, fill_value=0.0)

            new_data[mask, 1] = f(wavelength[mask])

            # Remplacer les données originales par les données interpolées
            setattr(self, dataset_name, new_data)

    @staticmethod
    def _QE2ApW(data: np.ndarray) -> np.ndarray:
        """
        Convertit l'efficacité quantique en ampères par watt

        :param data: Données des photodiodes
        :return: Données converties en ampères par watt
        """
        ApW_arr = np.zeros(np.shape(data))
        ApW_arr[:, 1] = data[:, 1] * constants.e / (constants.h * constants.c / (data[:, 0] * 1e-9))
        ApW_arr[:, 0] = data[:, 0]
        return ApW_arr

    @staticmethod
    def _photodiode_ADC(current_data, gain_transimp=1, ADC_max=4095, voltage_max=3.3):
        """
        Convertit le courant en valeurs ADC

        :param current_data: Données de courant
        :param gain_transimp: Gain du transimpédance (Default = 1)
        :param ADC_max: Valeur maximale de l'ADC (Default = 4095)
        :param voltage_max: Valeur maximale du voltage (Default = 3.3)
        :return: Données converties en valeurs ADC
        """
        # 4095*Voltage/3.3
        ADC_arr = np.zeros(np.shape(current_data))
        ADC_arr[:, 1] = current_data[:, 1] * gain_transimp * ADC_max / voltage_max
        ADC_arr[:, 0] = current_data[:, 0]
        return ADC_arr

    @staticmethod
    def _denormalize_curve(normalized_data, reference_wavelength, reference_value,
                          scaling_factor: float =1.0, sensor_area: float =1.0) -> np.ndarray:
        """
        Dénormalise une courbe en utilisant un point de référence.

        :param normalized_data: Données normalisées (format [longueur d'onde, valeur])
        :param reference_wavelength: Longueur d'onde de référence en nm
        :param reference_value: Valeur absolue à cette longueur d'onde (ex : counts/(µW/cm²))
        :param scaling_factor: Facteur d'échelle pour la dénormalisation (Default = 1)
        :param sensor_area: Aire du capteur (Default = 1)

        :return: Données dénormalisées (longueur d'onde, valeur dénormalisée)
        """
        # Créer une copie pour éviter de modifier les données originales
        denormalized_data = np.copy(normalized_data)

        # Trouver la valeur normalisée à la longueur d'onde de référence
        # Trouver l'indice le plus proche de la longueur d'onde de référence
        idx = np.abs(normalized_data[:, 0] - reference_wavelength).argmin()
        normalized_value_at_ref = normalized_data[idx, 1]

        # Si la valeur normalisée est 0, on ne peut pas calculer le facteur
        if normalized_value_at_ref == 0:
            print(f"Attention: La valeur normalisée à {reference_wavelength} nm est 0. Impossible de dénormaliser.")
            return denormalized_data

        # Calculer le facteur de mise à l'échelle
        scale_factor = reference_value / normalized_value_at_ref

        # Appliquer le facteur d'échelle à toutes les valeurs normalisées
        denormalized_data[:, 1] = normalized_data[:, 1] * scale_factor * scaling_factor / sensor_area

        return denormalized_data

    @staticmethod
    def _spectral_normalization(data_array):
        """
        Normalise les réponses des capteurs par rapport à la valeur maximale sur toute la plage de longueurs d'onde.
        Version vectorisée pour de meilleures performances.

        :param data_array: Liste des tableaux de données pour chaque capteur
        :return: Liste des tableaux de données normalisés
        """
        # Création d'une copie des données pour éviter de modifier les originales
        normalized_data = [np.copy(data) for data in data_array]

        # Extraction des valeurs (deuxième colonne) de chaque tableau
        values = np.array([data[:, 1] for data in normalized_data])

        # Calcul du maximum pour chaque point de longueur d'onde (sur tous les capteurs)
        max_values = np.max(values, axis=0)

        # Éviter la division par zéro en remplaçant les zéros par 1.
        max_values[max_values == 0] = 1

        # Normalisation de tous les capteurs en une seule opération
        for i, data in enumerate(normalized_data):
            data[:, 1] = values[i] / max_values

        return normalized_data

    def _process_data(self):
        """
        Traiter les données. Appliquer les méthodes de prétraitement nécessaires.
        """
        # Interpoler les données
        self._interpolate_data()

        # ------------ P_IR1 --------------
        P_IR1_interp = self._photodiode_ADC(self._QE2ApW(self.IR), self.dict_capteurs['P_IR1']['gain'])

        # ------------ P_IR1xP ------------
        P_IR1xP_interp = self.IR
        P_IR1xP_interp[:, 1] = self.IR[:, 1] * self.plastic_transmission[:, 1]
        P_IR1xP_interp = self._photodiode_ADC(self._QE2ApW(P_IR1xP_interp), self.dict_capteurs['P_IR1xP']['gain'])

        # ------------ P_IR2 --------------
        P_IR2_interp = self._photodiode_ADC(self._QE2ApW(self.IR2), self.dict_capteurs['P_IR2']['gain'])

        # ------------ P_UV ---------------
        P_UV_interp = self._photodiode_ADC(self._QE2ApW(self.UV2), self.dict_capteurs['P_UV']['gain'])


        # ------------ C_UV ---------------
        C_UV_interp = self._denormalize_curve(self.UV1, 310, 160/70,
                                        self.dict_capteurs['C_UV']['gain'],
                                        self.dict_capteurs['C_UV']['sensor_area'])

        # ------------ C_VISG -------------
        C_VISG_interp = abs(self._denormalize_curve(self.VIS_green, 518, 74,
                                              self.dict_capteurs['C_VISG']['gain'],
                                              self.dict_capteurs['C_VISG'][
                                                  'sensor_area']))

        # ------------ C_VISB -------------
        C_VISB_interp = self._denormalize_curve(self.VIS_Blue, 467, 56,
                                          self.dict_capteurs['C_VISB']['gain'],
                                          self.dict_capteurs['C_VISB']['sensor_area'])

        # ------------ C_VISR -------------
        C_VISR_interp = self._denormalize_curve(self.VIS_red, 619, 96,
                                          self.dict_capteurs['C_VISR']['gain'],
                                          self.dict_capteurs['C_VISR']['sensor_area'])

        # Ajouter les valeurs au dict_capteurs pour chaque capteur
        self.dict_capteurs['P_IR1']['data'] = P_IR1_interp
        self.dict_capteurs['P_IR1xP']['data'] = P_IR1xP_interp
        self.dict_capteurs['P_IR2']['data'] = P_IR2_interp
        self.dict_capteurs['P_UV']['data'] = P_UV_interp
        self.dict_capteurs['C_UV']['data'] = C_UV_interp
        self.dict_capteurs['C_VISG']['data'] = C_VISG_interp
        self.dict_capteurs['C_VISB']['data'] = C_VISB_interp
        self.dict_capteurs['C_VISR']['data'] = C_VISR_interp

        # Normaliser les données
        normalized_data_list= self._spectral_normalization(
            [P_IR1_interp, P_IR1xP_interp, P_IR2_interp, P_UV_interp,
             C_UV_interp,C_VISG_interp, C_VISB_interp, C_VISR_interp])

        P_IR1_N = normalized_data_list[0]
        P_IR1xP_N = normalized_data_list[1]
        P_IR2_N = normalized_data_list[2]
        P_UV_N = normalized_data_list[3]
        C_UV_N = normalized_data_list[4]
        C_VISG_N = normalized_data_list[5]
        C_VISB_N = normalized_data_list[6]
        C_VISR_N = normalized_data_list[7]

        self.all_sensors = {'P_IR1': P_IR1_N,
                            'P_IR1xP': P_IR1xP_N,
                            'P_IR2': P_IR2_N,
                            'P_UV': P_UV_N,
                            'C_UV': C_UV_N,
                            'C_VISG': C_VISG_N,
                            'C_VISB': C_VISB_N,
                            'C_VISR': C_VISR_N}

        self.Photodiodes_sensors = {'P_IR1': P_IR1_N,
                                    'P_IR1xP': P_IR1xP_N,
                                    'P_IR2': P_IR2_N,
                                    'P_UV': P_UV_N}

        self.I2C_sensors = {'C_UV': C_UV_N,
                            'C_VISG': C_VISG_N,
                            'C_VISB': C_VISB_N,
                            'C_VISR': C_VISR_N}

    @staticmethod
    def plot_response(responses):
        """
        Fonction pour afficher les courbes de réponse des capteurs
        """
        plt.figure(figsize=(8, 4))

        for sensor_name, data in responses.items():
            plt.plot(data[:, 0], data[:, 1], label=sensor_name)

        plt.title("Courbes de réponse des capteurs")
        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Réponse normalisée")
        plt.legend()
        plt.grid()
        plt.show()


donnees = DataPreProcess()
donnees.plot_response(donnees.all_sensors)
#print(donnees.UV1)

print(f"Temps d'exécution total : {perf_counter() - start_total_time}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")