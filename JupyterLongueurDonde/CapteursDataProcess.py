import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import constants
import matplotlib.pyplot as plt
from time import perf_counter



start_total_time = perf_counter()


class DataPreProcess:
    """
    Classe pour charger et traiter les données des capteurs
    """
    def __init__(self, callibration = None, callib_point='point1', plastic_name: str='Petri'):
        """
        Initialisation de l'objet des courbes de réponses

        :param gains: Liste des gains des capteurs. (Default = None)
        :param plastic_name: Nom du plastique pour la transmission. (Default = 'Petri')
        """

        if callibration is None:
            self.callibration = {'point1':{'puissance#W':1, 'longueur_donde#nm':450, 'counts':[17.6316, 15.7135, 8.9103, 236.3130, 9.5582, 232.6271, 1455.2121, 217.1186]},
                                 'point2':{'puissance#W':1, 'longueur_donde#nm':976, 'counts':[718.4668, 647.4318, 453.5327, 639.2649, 0.0000, 0.0000, 0.0000, 0.0000]},
                                 'point3':{'puissance#W':1, 'longueur_donde#nm':1976, 'counts':[2734.6400, 2214.5632, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]},
                                 'gains': [1, 1, 1, 1, 1, 1, 1, 1]}
            # ['P_IR1', 'P_IR1xP', 'P_IR2', 'P_UV', 'C_UV', 'C_VISG', 'C_VISB', 'C_VISR']
        else:
            self.callibration = callibration
        
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

        # Initialiser les positions des capteurs (x, y, z)
        self.P_IR1_pos = (-9.7, 21.34, -23.16)  # Position du capteur IR1 [mm]
        self.P_IR1xP_pos = (9.25, 21.18, -23.238)  # Position du capteur IR1xP [mm]
        self.P_IR2_pos = (0, 23.63, -20.66)  # Position du capteur IR2 [mm]
        self.P_UV_pos = (-0.63, 20.56, -26.89)  # Position du capteur UV [mm]
        self.C_UV_pos = (-5.36, 26.86, -30.14)  # Position du capteur UV I2C [mm]
        self.C_VISG_pos = (5.97, 16.97, -29.91)  # Position du capteur VIS I2C Green [mm]
        self.C_VISB_pos = (5.97, 16.97, -29.91)  # Position du capteur VIS I2C Blue [mm]
        self.C_VISR_pos = (5.97, 16.97, -29.91)  # Position du capteur VIS I2C Red [mm]
        
        # Utiliser les gains fournis
        self.gain_IR1 = self.callibration['gains'][0]
        self.gain_IR1xP = self.callibration['gains'][1]
        self.gain_IR2 = self.callibration['gains'][2]
        self.gain_UV = self.callibration['gains'][3]
        self.gain_C_UV = self.callibration['gains'][4]
        self.gain_C_VISG = self.callibration['gains'][5]
        self.gain_C_VISB = self.callibration['gains'][6]
        self.gain_C_VISR = self.callibration['gains'][7]

        self.sensor_order = ['P_IR1', 'P_IR1xP', 'P_IR2', 'P_UV', 'C_UV', 'C_VISG', 'C_VISB', 'C_VISR']

        # Créé un dictionnaire pour les capteurs
        self.dict_capteurs = {
            'P_IR1': {'gain': self.gain_IR1, 'position': self.P_IR1_pos, 'geo_factor': [1, 1, 1],},
            'P_IR1xP': {'gain': self.gain_IR1xP, 'position': self.P_IR1xP_pos, 'geo_factor': [1, 1, 1],},
            'P_IR2': {'gain': self.gain_IR2, 'position': self.P_IR2_pos, 'geo_factor': [1, 1, 1],},
            'P_UV': {'gain': self.gain_UV, 'position': self.P_UV_pos, 'geo_factor': [1, 1, 1],},
            'C_UV': {'gain': self.gain_C_UV, 'position': self.C_UV_pos, 'geo_factor': [1, 1, 1],},
            'C_VISG': {'gain': self.gain_C_VISG, 'position': self.C_VISG_pos, 'geo_factor': [1, 1, 1],},
            'C_VISB': {'gain': self.gain_C_VISB, 'position': self.C_VISB_pos, 'geo_factor': [1, 1, 1],},
            'C_VISR': {'gain': self.gain_C_VISR, 'position': self.C_VISR_pos, 'geo_factor': [1, 1, 1],}}
        
        
        
        # Charger des données
        self._load_data()

        # Charger les données expérimentales de dépendance angulaire
        self.angular_dict = self.variation_angulaire()
        
        # Définir le facteur de correction de référence pour chaque capteur
        self.set_ref_geo_factor()

        # Traiter les données
        self._process_data(point=callib_point)
        
    def set_ref_geo_factor(self, faisceau_pos=(0,0,0)):
        """
        Fonction pour calculer le facteur angulaire des capteurs
        """
        # Parcourir tous les capteurs et calculer les angles
        for sensor_name in self.sensor_order:
            f_x, f_y, f_z = faisceau_pos
            p_x, p_y, p_z = self.dict_capteurs[sensor_name]['position']
            
            # Calcul de l'angle entre le faisceau et le capteur
            angle = (180/np.pi) * (np.arccos(abs(p_z - f_z) / 
                                            np.sqrt((f_x - p_x)**2 + (f_y - p_y)**2 + (f_z - p_z)**2)))
            
            distance = np.sqrt((f_x - p_x)**2 + (f_y - p_y)**2 + (f_z - p_z)**2)
            
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
            
            # Calculer le facteur géométrique
            geo_factor = []
            for i in range(3):
                if i == 0:
                    geo_factor.append(intensite_450 / (distance**2))
                elif i == 1:
                    geo_factor.append(intensite_976 / (distance**2))
                elif i == 2:
                    geo_factor.append(intensite_1976 / (distance**2))

            self.dict_capteurs[sensor_name]['geo_factor'] = geo_factor
            
        

    def variation_angulaire(self):
        # Dictionnaire pour stocker les résultats
        resultats = {}
        
        # Préparation des données et interpolation pour les 3 courbes
        # 450nm
        angle_450 = 90 - np.array([75, 70, 65, 60, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37])
        intensite_rel_450 = [37.5, 34.8, 31.1, 27.8, 24.5, 24, 23.2, 22.6, 22.2, 21.7, 21.2, 20.5, 20, 19.4, 18.9, 18.3, 18.1, 17.5, 17.2, 16.6, 16.2, 15.7, 15.3]
        intensite_rel_450 = np.array(intensite_rel_450) / np.max(intensite_rel_450)
        
        # 976nm
        angle_976 = 90 - np.array([75, 70, 65, 60, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35])
        intensite_rel_976 = [56.8, 54.3, 51.2, 48.4, 44.4, 43.4, 43.2, 42.6, 41.3, 40.7, 40.1, 38.7, 38, 37.3, 36.5, 35.5, 34.7, 34, 33.2, 32.8, 32, 31.1, 30.5, 29.4, 29]
        intensite_rel_976 = np.array(intensite_rel_976) / np.max(intensite_rel_976)
        
        # 1976nm
        tension_ref = 148
        angle_1976 = 90 - np.array([75, 71, 69, 64, 59, 54, 49, 44, 39, 34, 29, 24])
        tension_mesuree = np.array([66, 70, 72, 73, 80, 82, 85, 89, 93, 101, 106, 110])
        intensite_rel_1976 = (tension_ref / tension_mesuree) / np.max(tension_ref / tension_mesuree)
        
        # Interpolation pour avoir 100 points pour chaque courbe
        # Déterminer les plages d'angle pour chaque courbe
        min_angle_450, max_angle_450 = min(angle_450), max(angle_450)
        min_angle_976, max_angle_976 = min(angle_976), max(angle_976)
        min_angle_1976, max_angle_1976 = min(angle_1976), max(angle_1976)
        
        angle_min = max(min_angle_450, min_angle_976, min_angle_1976)
        angle_max = min(max_angle_450, max_angle_976, max_angle_1976)
        
        # Créer des grilles de 100 points pour chaque courbe
        angles = np.linspace(angle_min, angle_max, 100)
        
        # Interpolation des valeurs d'intensité relative
        from scipy.interpolate import interp1d
        
        f_450 = interp1d(angle_450, intensite_rel_450, kind='cubic', bounds_error=False, fill_value='extrapolate')
        f_976 = interp1d(angle_976, intensite_rel_976, kind='cubic', bounds_error=False, fill_value='extrapolate')
        f_1976 = interp1d(angle_1976, intensite_rel_1976, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        intensite_450_interp = f_450(angles)
        intensite_976_interp = f_976(angles)
        intensite_1976_interp = f_1976(angles)
        
        # Parcourir tous les capteurs et calculer les angles
        for sensor_name in self.sensor_order:       
            # Stocker les résultats
            resultats[sensor_name] = {
                'angles': angles,
                'intensite_450nm': intensite_450_interp,
                'intensite_976nm': intensite_976_interp,
                'intensite_1976nm': intensite_1976_interp
            }
        
        return resultats

        
        
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
    def _denormalize_curve(normalized_data, reference_wavelength, reference_value,
                          scaling_factor: float =1.0) -> np.ndarray:
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
        denormalized_data[:, 1] = normalized_data[:, 1] * scale_factor * scaling_factor

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

    def _process_data(self, point):
        """
        Traiter les données. Appliquer les méthodes de prétraitement nécessaires.
        """
        # Interpoler les données
        self._interpolate_data()

        ref_wavelength = self.callibration[point]['longueur_donde#nm']

        # ------------ P_IR1 --------------
        P_IR1_interp = self._QE2ApW(self.IR)
        P_IR1_interp[:, 1] = P_IR1_interp[:, 1]/np.max(P_IR1_interp[:, 1])
        
        ref_value = self.callibration[point]['counts'][0] / self.callibration[point]['puissance#W'] 
        
        P_IR1_interp = self._denormalize_curve(P_IR1_interp, 
                                               ref_wavelength, 
                                               ref_value,
                                               self.dict_capteurs['P_IR1']['gain'])

        # ------------ P_IR1xP ------------
        P_IR1xP_interp = self.IR
        P_IR1xP_interp[:, 1] = self.IR[:, 1] * self.plastic_transmission[:, 1]
        P_IR1xP_interp = self._QE2ApW(P_IR1xP_interp)
        P_IR1xP_interp[:, 1] = P_IR1xP_interp[:, 1]/np.max(P_IR1xP_interp[:, 1])

        ref_value = self.callibration[point]['counts'][1] / self.callibration[point]['puissance#W']
        
        P_IR1xP_interp = self._denormalize_curve(P_IR1xP_interp,
                                                  ref_wavelength, 
                                                  ref_value,
                                                  self.dict_capteurs['P_IR1xP']['gain'])

        # ------------ P_IR2 --------------
        P_IR2_interp = self._QE2ApW(self.IR2)
        P_IR2_interp[:, 1] = P_IR2_interp[:, 1]/np.max(P_IR2_interp[:, 1])
        
        ref_value = self.callibration[point]['counts'][2] / self.callibration[point]['puissance#W']
        
        P_IR2_interp = self._denormalize_curve(P_IR2_interp,
                                               ref_wavelength, 
                                               ref_value,
                                               self.dict_capteurs['P_IR2']['gain'])
        

        # ------------ P_UV ---------------
        P_UV_interp = self._QE2ApW(self.UV2)
        P_UV_interp[:, 1] = P_UV_interp[:, 1]/np.max(P_UV_interp[:, 1])
        
        ref_value = self.callibration[point]['counts'][3] / self.callibration[point]['puissance#W']
        
        P_UV_interp = self._denormalize_curve(P_UV_interp,
                                              ref_wavelength, 
                                              ref_value,
                                              self.dict_capteurs['P_UV']['gain'])


        # ------------ C_UV ---------------
        ref_value = self.callibration[point]['counts'][4] / self.callibration[point]['puissance#W']
        
        C_UV_interp = self._denormalize_curve(self.UV1, 
                                              ref_wavelength, ref_value,
                                              self.dict_capteurs['C_UV']['gain'])

        # ------------ C_VISG -------------
        ref_value = self.callibration[point]['counts'][5] / self.callibration[point]['puissance#W']
        
        C_VISG_interp = abs(self._denormalize_curve(self.VIS_green, 
                                                    ref_wavelength, 
                                                    ref_value,
                                                    self.dict_capteurs['C_VISG']['gain']))

        # ------------ C_VISB -------------
        ref_value = self.callibration[point]['counts'][6] / self.callibration[point]['puissance#W']
        
        C_VISB_interp = self._denormalize_curve(self.VIS_Blue, 
                                                ref_wavelength, 
                                                ref_value,
                                                self.dict_capteurs['C_VISB']['gain'])
        
        # ------------ C_VISR -------------
        ref_value = self.callibration[point]['counts'][7] / self.callibration[point]['puissance#W']
        
        C_VISR_interp = self._denormalize_curve(self.VIS_red, 
                                                ref_wavelength, 
                                                ref_value,
                                                self.dict_capteurs['C_VISR']['gain'])

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

