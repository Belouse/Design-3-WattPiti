import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import constants 
import os
import pickle
import warnings
import time

class EntrainementLambda():

    def __init__(self, gains=[], plastic_name='Petri'):
        """
        Initialisation de l'objet des courbes de réponses 
        
        gains: liste contenant les gains de chaque capteur: 
        [P_IR1, P_IR1xP, P_IR2, P_UV, C_UV, C_VISG, C_VISB, C_VISR]
        
        P_IR1   : Photodiode IR 2500
        P_IR1xP : Photodiode IR 2500 PMMA
        P_IR2   : Photodiode IR 1700
        P_UV    : Photodiode UV
        C_UV    : Capteur I2C UV
        C_VISG  : Capteur I2C VIS channel Green
        C_VISB  : Capteur I2C VIS channel Blue
        C_VISR  : Capteur I2C VIS channel Red
        
        """
        
        # A corriger !!! Calcul de l'aire de la surface des capteurs I2C pour convertir en Counts/W
        P_IR1_area = 2
        P_IR1xP_area = 2 
        P_IR2_area = 2
        P_UV_area = 2
        C_UV_area = (0.28E-3)**2
        C_VISG_area = (0.2E-3)**2
        C_VISB_area = (0.2E-3)**2
        C_VISR_area = (0.2E-3)**2
        
                
        if gains == []:
            # Valeurs par défaut si aucun gain n'est fourni
            gain_IR1 = 2                        # Gain de la photodiode IR 2500
            gain_IR1xP = 2                      # Gain de la photodiode IR 2500 PMMA
            gain_IR2 = 2                        # Gain de la photodiode IR 1700
            gain_UV = 2                         # Gain de la photodiode UV
            gain_C_UV = 100 / 4e9 * 4095        # Gain du capteur I2C UV
            gain_C_VISG = 100 / 4e11 * 4095     # Gain du capteur I2C VIS channel Green
            gain_C_VISB = 100 / 4e11 * 4095     # Gain du capteur I2C VIS channel Blue
            gain_C_VISR = 100 / 4e11 * 4095     # Gain du capteur I2C VIS channel Red
        else:
            # Utiliser les gains fournis
            gain_IR1 = gains[0]                  # Gain de la photodiode IR 2500
            gain_IR1xP = gains[1]                # Gain de la photodiode IR 2500 PMMA
            gain_IR2 = gains[2]                  # Gain de la photodiode IR 1700
            gain_UV = gains[3]                   # Gain de la photodiode UV
            gain_C_UV = gains[4]                 # Gain du capteur I2C UV
            gain_C_VISG = gains[5]               # Gain du capteur I2C VIS channel Green
            gain_C_VISB = gains[6]               # Gain du capteur I2C VIS channel Blue
            gain_C_VISR = gains[7]               # Gain du capteur I2C VIS channel Red
                        
            
        self.dict_capteurs = {'P_IR1': {'gain': gain_IR1, 'sensor_area': P_IR1_area, 'data': []},
                              'P_IR1xP': {'gain': gain_IR1xP, 'sensor_area': P_IR1xP_area, 'data': []}, 
                              'P_IR2': {'gain': gain_IR2, 'sensor_area': P_IR2_area, 'data': []}, 
                              'P_UV': {'gain': gain_UV, 'sensor_area': P_UV_area, 'data': []}, 
                              'C_UV': {'gain': gain_C_UV, 'sensor_area': C_UV_area, 'data': []}, 
                              'C_VISG': {'gain': gain_C_VISG, 'sensor_area': C_VISG_area, 'data': []}, 
                              'C_VISB': {'gain': gain_C_VISB, 'sensor_area': C_VISB_area, 'data': []}, 
                              'C_VISR': {'gain': gain_C_VISR, 'sensor_area': C_VISR_area, 'data': []}}
                
        def interpolate_data(data, new_length=10000):
            # Récupérer les limites du fichier CSV
            min_x = data.iloc[:, 0].min()
            max_x = data.iloc[:, 0].max()
            
            # Créer le nouveau tableau d'interpolation
            new_data = np.zeros((new_length, 2))
            new_data[:, 0] = np.linspace(250, 2500, new_length)
            
            # Interpoler uniquement dans la plage valide et laisser à 0 ailleurs
            mask_in_range = (new_data[:, 0] >= min_x) & (new_data[:, 0] <= max_x)
            new_data[mask_in_range, 1] = np.interp(
                new_data[mask_in_range, 0], 
                data.iloc[:, 0], 
                data.iloc[:, 1]
            )
            # Les valeurs en dehors de la plage restent à 0 (valeur par défaut)
            
            return new_data

        def QE2ApW(data):
            ApW_arr = np.zeros(np.shape(data))
            ApW_arr[:, 1] = data[:, 1] * constants.e / (constants.h * constants.c / (data[:, 0] * 1e-9))
            ApW_arr[:, 0] = data[:, 0]
            return ApW_arr

        def photodiode_ADC(current_data, gain_transimp=1, ADC_max=4095, voltage_max=3.3):
            # 4095*Voltage/3.3
            ADC_arr = np.zeros(np.shape(current_data))
            ADC_arr[:, 1] = current_data[:, 1] * gain_transimp * ADC_max / voltage_max
            ADC_arr[:, 0] = current_data[:, 0]
            return ADC_arr

        def denormalize_curve(normalized_data, reference_wavelength, reference_value, scaling_factor = 1, sensor_area = 1):
            """
            Dénormalise une courbe en utilisant un point de référence.
            
            Parameters:
            normalized_data (numpy.ndarray): Données normalisées (format [longueur d'onde, valeur])
            reference_wavelength (float): Longueur d'onde de référence en nm
            reference_value (float): Valeur absolue à cette longueur d'onde (ex: counts/(µW/cm²))
            
            Returns:
            numpy.ndarray: Données dénormalisées avec la même forme que l'entrée
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

        def spectral_normalization(data_array):
            """
            Normalise les réponses des capteurs par rapport à la valeur maximale sur toute la plage de longueurs d'onde.
            Pour chaque longueur d'onde, trouve le capteur avec la réponse maximale et normalise les autres par rapport à celui-ci.
            
            Parameters:
            data_array (list): Liste des tableaux de données pour chaque capteur
            
            Returns:
            list: Liste des tableaux de données normalisés
            """
            # Création d'une copie des données pour éviter de modifier les originales
            normalized_data = [np.copy(data) for data in data_array]
            
            # Nombre de points de données (supposant que tous les capteurs ont le même nombre de points)
            num_points = normalized_data[0].shape[0]
            
            # Pour chaque point de longueur d'onde
            for i in range(num_points):
                # Collecter les valeurs de tous les capteurs à cette longueur d'onde
                values_at_wavelength = []
                for data in normalized_data:
                    values_at_wavelength.append(data[i, 1])
                
                # Trouver la valeur maximale parmi tous les capteurs à cette longueur d'onde
                max_value = max(values_at_wavelength)
                
                # Normaliser chaque capteur par rapport à la valeur maximale à cette longueur d'onde
                if max_value > 0:  # Éviter la division par zéro
                    for j in range(len(normalized_data)):
                        normalized_data[j][i, 1] = normalized_data[j][i, 1] / max_value
            
            return normalized_data

        # Charger les fichiers CSV
        # Obtenir le chemin du répertoire contenant le script en cours d'exécution
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Charger les fichiers CSV avec des chemins relatifs au script
        UV1 = pd.read_csv(os.path.join(script_dir, "UVS.csv"))
        UV2 = pd.read_csv(os.path.join(script_dir, "UV2.csv"))
        VIS_green = pd.read_csv(os.path.join(script_dir, "Green.csv"))
        VIS_Blue = pd.read_csv(os.path.join(script_dir, "Blue.csv"))
        VIS_red = pd.read_csv(os.path.join(script_dir, "Red.csv"))
        IR = pd.read_csv(os.path.join(script_dir, "IR.csv"))
        IR2 = pd.read_csv(os.path.join(script_dir, "IR2.csv"))
        platic_t_file = pd.read_csv(os.path.join(script_dir, "TransmissionsPlastiques.csv"))     
        
        # Extraction des données de transmission du plastique de petri
        plastic_index = platic_t_file.columns.get_loc(plastic_name)  # Trouver l'index de la colonne correspondant au plastique
        plastic_transmission = platic_t_file.iloc[1:, plastic_index + 1].values.astype(np.float64)[::-1] / 100
        wavelength = platic_t_file.iloc[1:, plastic_index].values.astype(np.float64)[::-1]
        plastic_transmission = pd.DataFrame(np.column_stack((wavelength, plastic_transmission)))
    
        # Interpolation des données des photodiodes et conversion en Counts/W
        P_IR1_interp =  photodiode_ADC(QE2ApW(interpolate_data(IR)), self.dict_capteurs['P_IR1']['gain'])
                                              
        PMMA_interp = interpolate_data(plastic_transmission)                                      
        P_IR1xP_interp =  interpolate_data(IR)
        P_IR1xP_interp[:, 1] = P_IR1xP_interp[:, 1] * PMMA_interp[:, 1]
        P_IR1xP_interp = photodiode_ADC(QE2ApW(P_IR1xP_interp), self.dict_capteurs['P_IR1xP']['gain'])
    
        P_IR2_interp = photodiode_ADC(QE2ApW(interpolate_data(IR2)), self.dict_capteurs['P_IR2']['gain'])      
        P_UV_interp = photodiode_ADC(QE2ApW(interpolate_data(UV2)), self.dict_capteurs['P_UV']['gain'])

        # Interpolation et dénormalisation des données des capteurs I2C
        C_UV_interp = denormalize_curve(interpolate_data(UV1), 310, 160/70, self.dict_capteurs['C_UV']['gain'], self.dict_capteurs['C_UV']['sensor_area'])
        C_VISG_interp = abs(denormalize_curve(interpolate_data(VIS_green), 518, 74, self.dict_capteurs['C_VISG']['gain'] , self.dict_capteurs['C_VISG']['sensor_area']))
        C_VISB_interp = denormalize_curve(interpolate_data(VIS_Blue), 467, 56, self.dict_capteurs['C_VISB']['gain'], self.dict_capteurs['C_VISB']['sensor_area'])
        C_VISR_interp = denormalize_curve(interpolate_data(VIS_red), 619, 96, self.dict_capteurs['C_VISR']['gain'], self.dict_capteurs['C_VISR']['sensor_area'])
            
        self.dict_capteurs['P_IR1']['data'] = P_IR1_interp
        self.dict_capteurs['P_IR1xP']['data'] = P_IR1xP_interp
        self.dict_capteurs['P_IR2']['data'] = P_IR2_interp
        self.dict_capteurs['P_UV']['data'] = P_UV_interp
        self.dict_capteurs['C_UV']['data'] = C_UV_interp
        self.dict_capteurs['C_VISG']['data'] = C_VISG_interp
        self.dict_capteurs['C_VISB']['data'] = C_VISB_interp
        self.dict_capteurs['C_VISR']['data'] = C_VISR_interp
                    
        P_IR1_N, P_IR1xP_N, P_IR2_N, P_UV_N, C_UV_N, C_VISG_N, C_VISB_N, C_VISR_N = spectral_normalization([P_IR1_interp, P_IR1xP_interp, P_IR2_interp, P_UV_interp, C_UV_interp, C_VISG_interp, C_VISB_interp, C_VISR_interp])
        
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
            
    
    def plot_response(self, repsonses):
        """
        Fonction pour afficher les courbes de réponse des capteurs
        
        all_sensors.plot_response()
        Photodiodes_sensors.plot_response()
        I2C_sensors.plot_response()
        
        """
        
        plt.figure(figsize=(8, 4))
        
        for sensor_name, data in repsonses.items():
            plt.plot(data[:, 0], data[:, 1], label=sensor_name)
        
        plt.title("Courbes de réponse des capteurs")
        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Réponse normalisée")
        plt.legend()
        plt.grid()
        plt.show()
    
    def train(self, training_responses, display_interval=1000, save_model=False, save_path="model_lambda.pkl"):
        """
        Entraîne un réseau de neurones sur les réponses des capteurs pour prédire la longueur d'onde.
        
        Parameters:
        training_responses (dict): Dictionnaire contenant les réponses des capteurs
        display_interval (int): Intervalle pour afficher le graphique pendant l'entraînement (en nombre d'itérations)
        save_model (bool): Si True, sauvegarde le modèle entraîné
        save_path (str): Chemin où sauvegarder le modèle
        
        Returns:
        MLPRegressor: Le modèle entraîné
        """
        
        
        training_data = []    
    
        for sensor_name, data in training_responses.items():
            training_data.append(data[:, 1])  # On prend seulement la colonne des valeurs            
        
        X = np.vstack(training_data).T  # Mise en forme pour sklearn (chaque ligne = 1 échantillon)

        # On prend le premier élément du dictionnaire comme référence pour la longueur d'onde
        first_key = list(training_responses.keys())[0]
        y = training_responses[first_key][:, 0]  # Longueur d'onde (première colonne)

        # Séparation des données en train/test (80% entraînement, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Création d'un MLPRegressor avec warm_start pour pouvoir continuer l'entraînement
        mlp = MLPRegressor(
            hidden_layer_sizes=(1000, 1000),    # Plus de couches et neurones
            activation='relu',                  # 'relu' fonctionne souvent mieux que 'tanh'
            solver='adam',                      # 'adam' fonctionne bien pour des datasets de taille moyenne
            alpha=0.001,                        # Régularisation L2 pour éviter l'overfitting
            learning_rate='adaptive',           # Ajuste le taux d'apprentissage si le modèle stagne
            max_iter=display_interval,          # Entraînement par étapes
            warm_start=True,                    # Permet de continuer l'entraînement
            random_state=42
        )
        
        max_iterations = 10000
        total_iterations = 0
        
        # Entraînement par étapes avec affichage des graphiques intermédiaires
        while total_iterations < max_iterations:
            start_time = time.time()
            
            # Désactiver les avertissements de non-convergence
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                mlp.fit(X_train, y_train)
            
            total_iterations += display_interval
            
            # Prédiction sur les données de test
            y_pred = mlp.predict(X_test)
            
            # Évaluation des performances
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculer le temps écoulé
            elapsed_time = time.time() - start_time
            
            print(f"Itération {total_iterations}/{max_iterations} - Temps écoulé: {elapsed_time:.2f}s")
            print(f"Erreur absolue moyenne (MAE) : {mae:.2f} nm")
            print(f"Coefficient de détermination (R²) : {r2:.3f}")
            
            # Tracer les résultats intermédiaires
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, s=5, alpha=0.6, label="Prédictions vs Réel") 
            plt.plot([250, 2500], [250, 2500], 'r--', label="Idéal (y = x)")
            plt.title(f"Résultats après {total_iterations} itérations - MAE: {mae:.2f}nm - R²: {r2:.3f}")
            plt.xlabel("Longueur d'onde réelle [nm]")
            plt.ylabel("Longueur d'onde prédite [nm]")
            plt.legend()
            plt.grid()
            plt.savefig
            plt.show()
            
            # Si on a atteint la performance désirée, on peut s'arrêter
            if r2 > 0.995:  # Exemple de critère d'arrêt
                print("Convergence atteinte avec R² > 0.99. Arrêt de l'entraînement.")
                break
            
            # Continuer l'entraînement pour un autre intervalle
            mlp.max_iter += display_interval
            
            # Si l'utilisateur veut sauvegarder des modèles intermédiaires
            if save_model and total_iterations % (display_interval * 5) == 0:
                interim_save_path = save_path.replace('.pkl', f'_iter{total_iterations}.pkl')
                with open(interim_save_path, 'wb') as f:
                    pickle.dump(mlp, f)
                print(f"Modèle intermédiaire sauvegardé à {interim_save_path}")
        
        # Sauvegarde du modèle final si demandé
        if save_model:
            with open(save_path, 'wb') as f:
                pickle.dump(mlp, f)
            print(f"Modèle final sauvegardé à {save_path}")

        # Graphique final
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, s=5, alpha=0.6, label="Prédictions vs Réel") 
        plt.plot([250, 2500], [250, 2500], 'r--', label="Idéal (y = x)")
        plt.title(f"Résultat final - MAE: {mae:.2f}nm - R²: {r2:.3f}")
        plt.xlabel("Longueur d'onde réelle [nm]")
        plt.ylabel("Longueur d'onde prédite [nm]")
        plt.legend()
        plt.grid()
        plt.savefig("resultat_final.png")
        plt.show()

        return mlp

if __name__ == "__main__":
 
    entrainement = EntrainementLambda()
    
    all_responses = entrainement.all_sensors
    photo_repsonses = entrainement.Photodiodes_sensors
    I2C_repsonses = entrainement.I2C_sensors
    
    # Afficher les courbes de réponse des capteurs
    # entrainement.plot_response(all_responses)
    # entrainement.plot_response(photo_repsonses)
    # entrainement.plot_response(I2C_repsonses)
    
    # Entraîner le modèle avec affichage tous les 1000 itérations et sauvegarde du modèle
    NN_model = entrainement.train(
        all_responses,
        display_interval=1000,
        save_model=True,
        save_path="modele_lambda_final.pkl"
    )
    
    
        
#     def train(self, training_responses):
#         """
#         Entraîne un réseau de neurones sur les réponses des capteurs pour prédire la longueur d'onde.
#         """
#         training_data = []    
    
#         for sensor_name, data in training_responses.items():
#             training_data.append(data[:, 1])  # On prend seulement la colonne des valeurs            
        
#         X = np.vstack(training_data).T  # Mise en forme pour sklearn (chaque ligne = 1 échantillon)

        
#         # On prend la première le premier élément du dictionnaire comme référence pour la longueur d'onde
#         first_key = list(training_responses.keys())[0]
#         y = training_responses[first_key][:, 0]  # Longueur d'onde (première colonne)

#         # Séparation des données en train/test (80% entraînement, 20% test)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         mlp = MLPRegressor(
#             hidden_layer_sizes=(1000, 1000),    # Plus de couches et neurones
#             activation='relu',                  # 'relu' fonctionne souvent mieux que 'tanh'
#             solver='adam',                      # 'adam' fonctionne bien pour des datasets de taille moyenne
#             alpha=0.001,                        # Régularisation L2 pour éviter l’overfitting
#             learning_rate='adaptive',           # Ajuste le taux d'apprentissage si le modèle stagne
#             max_iter=10000,                     # Plus d’itérations pour améliorer la convergence
#             random_state=42
#         )

#         mlp.fit(X_train, y_train)

#         # Prédiction sur les données de test
#         y_pred = mlp.predict(X_test)

#         # Évaluation des performances
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         print(f"Erreur absolue moyenne (MAE) : {mae:.2f} nm")
#         print(f"Coefficient de détermination (R²) : {r2:.3f}")

#         # Tracer les résultats
#         plt.scatter(y_test, y_pred, s=5, alpha=0.6, label="Prédictions vs Réel") 
#         plt.plot([250, 2500], [250, 2500], 'r--', label="Idéal (y = x)")
#         plt.xlabel("Longueur d'onde réelle [nm]")
#         plt.ylabel("Longueur d'onde prédite [nm]")
#         plt.legend()
#         plt.grid()
#         plt.show()

#         wavelength = 1000

#         return wavelength

# if __name__ == "__main__":
 
#     entrainement = EntrainementLambda()
    
#     all_responses = entrainement.all_sensors
#     photo_repsonses = entrainement.Photodiodes_sensors
#     I2C_repsonses = entrainement.I2C_sensors
    
#     # entrainement.plot_response(all_responses)
#     # entrainement.plot_response(photo_repsonses)
#     # entrainement.plot_response(I2C_repsonses)
    
#     NN_model = entrainement.train(all_responses)

    
    
   