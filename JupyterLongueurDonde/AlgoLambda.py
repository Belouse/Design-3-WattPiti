import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
import os

from EntrainementLambda import EntrainementLambda
from CapteursDataProcess import DataPreProcess
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import torch


from NN_Pytorch_Lambda import WavelengthPredictor


class AlgoWavelength:
    """
    Classe pour charger et utiliser un modèle entraîné pour prédire la longueur d'onde
    à partir des valeurs des capteurs.
    """

    def __init__(self, model_path='heavyModel.pkl'):
        """
        Initialise l'algorithme de prédiction de longueur d'onde en chargeant le modèle préentraîné.
        
        Parameters:
        model_path (str): Chemin vers le fichier du modèle entraîné
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
            #with open(model_path_abs, 'rb') as f:
            #    self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")
        
        # Stocker l'ordre des capteurs pour référence
        self.sensor_order = ['P_IR1', 'P_IR1xP', 'P_IR2', 'P_UV', 'C_UV', 'C_VISG', 'C_VISB', 'C_VISR']

        # Offset pour mise à zéro
        self.zero_offset = np.zeros(len(self.sensor_order))

        # Initialiser les sensor values
        self.sensor_values = np.zeros(len(self.sensor_order))
        
        # Créer une instance de EntrainementLambda pour avoir accès aux réponses des capteurs
        self.data_preprocess = DataPreProcess()
        self.angular_dict = self.data_preprocess.angular_dict
        self.responses = self.data_preprocess.all_sensors
        self.response_dict = self.data_preprocess.dict_capteurs


    def angular_factor(self, faisceau_pos=(0,0,0)):
        # Parcourir tous les capteurs et calculer les angles
        geo_factor_list = []

        for sensor_name in self.sensor_order:
            f_x, f_y, f_z = faisceau_pos
            p_x, p_y, p_z = self.response_dict[sensor_name]['position']

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

            ref_factor_450, ref_factor_976, ref_factor_1976  = self.response_dict[sensor_name]['geo_factor']

            # Calculer le facteur géométrique
            geo_factor = []
            for i in range(3):
                if i == 0:
                    geo_factor.append(ref_factor_450 / (intensite_450 / (distance**2)))
                elif i == 1:
                    geo_factor.append(ref_factor_976 / (intensite_976 / (distance**2)))
                elif i == 2:
                    geo_factor.append(ref_factor_1976 / (intensite_1976 / (distance**2)))

            geo_factor_list.append(geo_factor)

        return geo_factor_list


    def calculate_wavelength(self, sensor_values, faisceau_pos=(0,0,0), correction_factor_ind=0, enable_print=False):
        """
        Prédit la longueur d'onde à partir des valeurs des capteurs.
        
        Parameters:
        sensor_values (list or numpy.ndarray): Liste ou tableau des valeurs normalisées des capteurs
                                               dans l'ordre [P_IR1, P_IR1xP, P_IR2, P_UV, C_UV, C_VISG, C_VISB, C_VISR]
        
        Returns:
        float: Longueur d'onde prédite en nanomètres
        """
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


    def get_sensor_ratios_for_wavelength(self, wavelength, enable_print=False):
        """
        Extrait les ratios d'illumination de chaque capteur pour une longueur d'onde donnée.
        
        Parameters:
        wavelength (float): Longueur d'onde en nanomètres pour laquelle extraire les ratios
        entrainement_instance (EntrainementLambda, optional): Instance de la classe EntrainementLambda.
                                                              Si None, utilise l'instance interne.
        
        Returns:
        dict: Dictionnaire contenant les ratios d'illumination pour chaque capteur
        list: Liste des ratios dans l'ordre [P_IR1, P_IR1xP, P_IR2, P_UV, C_UV, C_VISG, C_VISB, C_VISR]
        """
    
        sensors_data = self.responses

        # Dictionnaire pour stocker les ratios
        ratios_dict = {}
        
        # Liste pour stocker les ratios dans l'ordre spécifié
        ratios_list = []
        
        # Pour chaque capteur, trouver la valeur à la longueur d'onde spécifiée
        for sensor_name in self.sensor_order:
            sensor_data = sensors_data[sensor_name]
            
            # Trouver l'indice le plus proche de la longueur d'onde demandée
            idx = np.abs(sensor_data[:, 0] - wavelength).argmin()
            
            # Obtenir la valeur du capteur à cette longueur d'onde
            ratio = sensor_data[idx, 1]
            
            # Stocker dans le dictionnaire et la liste
            ratios_dict[sensor_name] = ratio
            ratios_list.append(ratio)
        
        if enable_print:
            print("\nRatios des capteurs pour {} nm:".format(wavelength))
            for sensor, ratio in ratios_dict.items():
                print(f"{sensor}: {ratio:.4f}")
        
        return ratios_dict, ratios_list

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

    def test_model_with_wavelength(self, test_wavelength, entrainement_instance=None, enable_print=False):
        """
        Teste le modèle en simulant les valeurs des capteurs pour une longueur d'onde donnée,
        puis en utilisant ces valeurs pour prédire la longueur d'onde.
        
        Parameters:
        test_wavelength (float): Longueur d'onde à tester (en nm)
        entrainement_instance (EntrainementLambda, optional): Instance de la classe EntrainementLambda.
                                                             Si None, utilise l'instance interne.
        
        Returns:
        tuple: (longueur d'onde de test, longueur d'onde prédite, erreur absolue)
        """
        # Obtenir les ratios des capteurs pour la longueur d'onde de test
        _, sensor_ratios = self.get_sensor_ratios_for_wavelength(test_wavelength, entrainement_instance)
        
        # Utiliser le modèle pour prédire la longueur d'onde à partir des ratios
        predicted_wavelength = self.calculate_wavelength(sensor_ratios)
        
        # Calculer l'erreur absolue
        error = abs(predicted_wavelength - test_wavelength)
        
        if enable_print:
            print("\nTest du modèle:")
            print(f"Longueur d'onde de test: {test_wavelength} nm")
            print(f"Longueur d'onde prédite: {predicted_wavelength:.2f} nm")
            print(f"Erreur absolue: {error:.2f} nm")
    
        return test_wavelength, predicted_wavelength, error
    
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
        plt.ylabel("Réponse normalisée")
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

def test_noise_influence(wavelengths=[450, 976, 1976], noise_levels=np.arange(0, 0.21, 0.02), 
                        num_iterations=10, model_path='model_nn_pytorch_weights.pth'):
    """
    Teste l'influence du bruit sur la précision des prédictions de longueur d'onde.
    
    Parameters:
    wavelengths (list): Liste des longueurs d'onde à tester
    noise_levels (array): Niveaux de bruit à appliquer (proportion de la valeur)
    num_iterations (int): Nombre d'itérations pour chaque combinaison pour obtenir des statistiques
    model_path (str): Chemin vers le modèle entraîné
    
    Returns:
    tuple: (results, sensor_variations) 
           - results: dict contenant les pourcentages d'erreur moyens pour chaque combinaison
           - sensor_variations: dict contenant les variations moyennes de chaque capteur avec leur signe
    """
    # Initialiser l'algorithme de prédiction
    algo = AlgoWavelength(model_path=model_path)
    
    # Préparer la structure pour stocker les résultats
    results = {
        'wavelength': [],
        'noise_level': [],
        'error_percent': []
    }
    
    # Structure pour stocker les variations des réponses des capteurs
    sensor_variations = {
        'wavelength': [],
        'noise_level': [],
        'sensor_name': [],
        'variation_percent': []
    }
    
    # Pour chaque longueur d'onde
    for wavelength in wavelengths:
        print(f"Traitement de la longueur d'onde: {wavelength} nm")
        
        # Obtenir les réponses des capteurs pour cette longueur d'onde
        _, responses_list = algo.get_sensor_response_for_wavelength(wavelength)
        
        # Pour chaque niveau de bruit
        for noise_level in noise_levels:
            error_percents = []
            
            # Pour chaque capteur, initialiser une liste pour stocker les variations
            sensor_variations_this_level = [[] for _ in range(len(algo.sensor_order))]
            
            # Répéter plusieurs fois pour obtenir des statistiques
            for _ in range(num_iterations):
                # Ajouter du bruit aux réponses brutes (avant normalisation)
                noisy_responses = []
                for i, response in enumerate(responses_list):
                    # Générer un bruit aléatoire entre -noise_level et +noise_level de la valeur
                    noise = np.random.uniform(-noise_level, noise_level) * response
                    noisy_response = response + noise
                    noisy_response = max(0, noisy_response)  # Assurer des valeurs positives
                    
                    # Calculer la variation en pourcentage avec signe
                    if response > 0:  # Éviter division par zéro
                        # Garder le signe de la variation (positif = augmentation, négatif = diminution)
                        variation_percent = (noisy_response - response) / response * 100
                        sensor_variations_this_level[i].append(variation_percent)
                    
                    noisy_responses.append(noisy_response)
                
                # Normaliser les réponses avec bruit
                noisy_ratios = np.array(noisy_responses) / np.max(noisy_responses)
                
                # Prédire la longueur d'onde avec les ratios bruités
                predicted_wavelength = algo.calculate_wavelength(noisy_ratios)
                
                # Calculer l'erreur en pourcentage
                error_percent = abs(predicted_wavelength - wavelength) / wavelength * 100
                error_percents.append(error_percent)
            
            # Calculer la moyenne des erreurs pour ce niveau de bruit et cette longueur d'onde
            mean_error_percent = np.mean(error_percents)
            
            # Stocker les résultats d'erreur
            results['wavelength'].append(wavelength)
            results['noise_level'].append(noise_level)
            results['error_percent'].append(mean_error_percent)
            
            # Stocker les variations moyennes pour chaque capteur
            for i, variations in enumerate(sensor_variations_this_level):
                if variations:  # S'assurer qu'il y a des données
                    sensor_variations['wavelength'].append(wavelength)
                    sensor_variations['noise_level'].append(noise_level)
                    sensor_variations['sensor_name'].append(algo.sensor_order[i])
                    sensor_variations['variation_percent'].append(np.mean(variations))
    
    return results, sensor_variations

def plot_separate_heatmaps(results, sensor_variations):
    """
    Crée deux figures séparées:
    1. Une heatmap d'erreur de prédiction pour chaque longueur d'onde
    2. Une figure avec trois heatmaps empilées verticalement montrant la variation des capteurs pour chaque longueur d'onde
    
    Parameters:
    results (dict): Résultats d'erreur obtenus de la fonction test_noise_influence
    sensor_variations (dict): Variations des capteurs obtenues de la fonction test_noise_influence
    """
    # Convertir les résultats en DataFrame
    df_results = pd.DataFrame(results)
    df_sensors = pd.DataFrame(sensor_variations)
    
    # FIGURE 1: Heatmap d'erreur de prédiction
    plt.figure(figsize=(10, 6))
    
    # Créer un pivot pour la heatmap d'erreur
    pivot_error = df_results.pivot(index='noise_level', columns='wavelength', values='error_percent')
    
    # Créer la heatmap d'erreur
    ax = sns.heatmap(pivot_error, annot=True, fmt=".2f", cmap="YlOrRd", 
                cbar_kws={'label': 'Erreur (%)'})
    
    # Ajouter des titres et étiquettes
    plt.title("Erreur de prédiction par longueur d'onde", fontsize=14)
    plt.xlabel("Longueur d'onde (nm)", fontsize=12)
    plt.ylabel("Niveau de bruit", fontsize=12)
    
    # Améliorer les étiquettes de l'axe des x
    ax.set_xticklabels([f"{int(wl)} nm" for wl in pivot_error.columns])
    
    # Améliorer les étiquettes de l'axe des y
    ax.set_yticklabels([f"{nl:.2f}" for nl in pivot_error.index])
    
    plt.tight_layout()
    plt.savefig('heatmap_error_prediction.png', dpi=300)
    plt.show()
    
    # FIGURE 2: Heatmaps empilées des variations de capteurs pour chaque longueur d'onde
    wavelengths = df_results['wavelength'].unique()
    num_wavelengths = len(wavelengths)
    
    # Créer une figure plus haute pour accueillir les trois heatmaps empilées
    fig, axes = plt.subplots(num_wavelengths, 1, figsize=(10, 3 * num_wavelengths))
    
    # Définir une palette de couleurs divergente avec le point central à 0
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Pour chaque longueur d'onde
    for i, wavelength in enumerate(wavelengths):
        # Filtrer pour cette longueur d'onde spécifique
        df_sensors_filtered = df_sensors[df_sensors['wavelength'] == wavelength]
        
        # Créer un pivot pour la heatmap de variation des capteurs
        pivot_sensors = df_sensors_filtered.pivot(index='noise_level', columns='sensor_name', values='variation_percent')
        
        # Créer la heatmap de variation des capteurs sur l'axe correspondant
        ax = axes[i] if num_wavelengths > 1 else axes
        
        # Trouver la valeur absolue maximale pour centrer correctement la carte de couleurs
        vmax = max(abs(pivot_sensors.values.min()), abs(pivot_sensors.values.max()))
        
        # Créer la heatmap avec une coloration divergente (bleu = négatif, rouge = positif)
        sns.heatmap(pivot_sensors, annot=True, fmt="+.2f", cmap=cmap,
                    vmin=-vmax, vmax=vmax, center=0,
                    cbar_kws={'label': 'Variation (%)'}, ax=ax)
        
        # Ajouter des titres et étiquettes
        ax.set_title(f"Variation des réponses des capteurs pour λ = {wavelength} nm", fontsize=14)
        ax.set_xlabel("Capteur", fontsize=12)
        ax.set_ylabel("Niveau de bruit", fontsize=12)
        
        # Améliorer les étiquettes de l'axe des y
        ax.set_yticklabels([f"{nl:.2f}" for nl in pivot_sensors.index])
    
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    plt.savefig('heatmap_sensor_variations.png', dpi=300)
    plt.show()
    
def plot_wavelength_errors(results):
    """
    Crée un graphique montrant l'évolution de l'erreur en fonction du niveau de bruit
    pour chaque longueur d'onde testée.
    
    Parameters:
    results (dict): Résultats obtenus de la fonction test_noise_influence
    """
    # Convertir les résultats en DataFrame
    df = pd.DataFrame(results)
    
    # Créer la figure
    plt.figure(figsize=(10, 6))
    
    # Tracer les courbes pour chaque longueur d'onde
    for wavelength in df['wavelength'].unique():
        subset = df[df['wavelength'] == wavelength]
        plt.plot(subset['noise_level'], subset['error_percent'], 
                 marker='o', label=f"{wavelength} nm")
    
    # Ajouter des titres et étiquettes
    plt.title("Évolution de l'erreur en fonction du niveau de bruit")
    plt.xlabel("Niveau de bruit")
    plt.ylabel("Erreur (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('error_vs_noise.png')
    plt.show()
    
    def dependance_angulaire(self, wavelength, angle, enable_print=False):
        return

def map_position_error():
    """
    Fonction qui mappe l'erreur sur la longueur d'onde estimée en fonction de la position
    du faisceau pour les trois indices de correction (0, 1, 2).
    Génère trois heatmaps, une pour chaque indice, avec l'erreur en pourcentage.
    Les positions sont incluses dans un cercle de 25 de diamètre centré à l'origine.
    """
    algo = AlgoWavelength(model_path='model_nn_pytorch_weights.pth')
    
    # Définir les longueurs d'onde de test
    wavelengths = [450, 976, 1976]
    
    # Définir la grille de positions x,y
    radius = 25/2  # Rayon du cercle
    resolution = 21  # Nombre de points dans chaque direction
    
    # Créer les grilles de coordonnées x et y
    x_positions = np.linspace(-radius, radius, resolution)
    y_positions = np.linspace(-radius, radius, resolution)
    
    # Préparer les conteneurs pour les résultats
    results = {
        'correction_factor_ind': [],
        'wavelength': [],
        'x_pos': [],
        'y_pos': [],
        'error_percent': []
    }
    
    # Pour chaque indice de correction
    for correction_factor_ind in [0, 1, 2]:
        print(f"Traitement avec indice de correction: {correction_factor_ind}")
        
        # Pour chaque longueur d'onde
        for wavelength in wavelengths:
            print(f"  - Longueur d'onde: {wavelength} nm")
            
            # Obtenir les réponses des capteurs pour cette longueur d'onde
            _, responses_list = algo.get_sensor_response_for_wavelength(wavelength)
            ratios_list = np.array(responses_list) / np.max(responses_list)
            
            # Valeur de référence à la position (0,0,0)
            reference_wavelength = algo.calculate_wavelength(
                ratios_list.copy(), 
                faisceau_pos=(0, 0, 0), 
                correction_factor_ind=correction_factor_ind
            )
            
            # Pour chaque position x, y
            for x in x_positions:
                for y in y_positions:
                    # Vérifier si la position est dans le cercle
                    if x**2 + y**2 <= radius**2:
                        # Prédire la longueur d'onde avec la position du faisceau
                        predicted_wavelength = algo.calculate_wavelength(
                            ratios_list.copy(),
                            faisceau_pos=(x, y, 0),
                            correction_factor_ind=correction_factor_ind
                        )
                        
                        # Calculer l'erreur en pourcentage par rapport à la valeur à la position d'origine
                        # (pour isoler l'effet de la position)
                        error_percent = abs(predicted_wavelength - reference_wavelength) / reference_wavelength * 100
                        
                        # Stocker les résultats
                        results['correction_factor_ind'].append(correction_factor_ind)
                        results['wavelength'].append(wavelength)
                        results['x_pos'].append(x)
                        results['y_pos'].append(y)
                        results['error_percent'].append(error_percent)
    
    # Convertir les résultats en DataFrame
    df_results = pd.DataFrame(results)
    
    # Créer les heatmaps
    plot_position_error_heatmaps(df_results, radius, resolution)

def plot_position_error_heatmaps(df_results, radius, resolution):
    """
    Génère trois heatmaps montrant l'erreur en fonction de la position pour chaque indice de correction.
    
    Parameters:
    df_results (pandas.DataFrame): DataFrame contenant les résultats
    radius (float): Rayon du cercle de positions
    resolution (int): Résolution de la grille
    """
    # Obtenir les valeurs uniques des indices de correction et des longueurs d'onde
    correction_indices = sorted(df_results['correction_factor_ind'].unique())
    wavelengths = sorted(df_results['wavelength'].unique())
    
    # Créer une figure avec des sous-graphiques (1 ligne, 3 colonnes)
    fig, axes = plt.subplots(len(correction_indices), len(wavelengths), 
                            figsize=(15, 12), 
                            constrained_layout=True)
    
    # Configuration des niveaux de contour pour les heatmaps
    levels = np.linspace(0, df_results['error_percent'].max(), 20)
    
    # Créer un maillage de coordonnées pour le graphique
    x_grid = np.linspace(-radius, radius, resolution)
    y_grid = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Pour chaque indice de correction
    for i, correction_ind in enumerate(correction_indices):
        # Pour chaque longueur d'onde
        for j, wavelength in enumerate(wavelengths):
            # Filtrer les données pour cet indice de correction et cette longueur d'onde
            subset = df_results[(df_results['correction_factor_ind'] == correction_ind) & 
                               (df_results['wavelength'] == wavelength)]
            
            # Créer une grille pour la heatmap
            error_grid = np.zeros((resolution, resolution)) * np.nan
            
            # Remplir la grille avec les erreurs
            for _, row in subset.iterrows():
                # Trouver les indices de la grille correspondant aux positions x, y
                x_idx = np.abs(x_grid - row['x_pos']).argmin()
                y_idx = np.abs(y_grid - row['y_pos']).argmin()
                error_grid[y_idx, x_idx] = row['error_percent']
            
            # Masquer les points en dehors du cercle
            mask = X**2 + Y**2 > radius**2
            error_grid = np.ma.array(error_grid, mask=mask)
            
            # Tracer la heatmap dans le sous-graphique correspondant
            ax = axes[i, j]
            contour = ax.contourf(X, Y, error_grid, levels=levels, cmap='viridis', extend='max')
            
            # Ajouter un cercle pour montrer la limite
            circle = plt.Circle((0, 0), radius, fill=False, color='r', linestyle='--')
            ax.add_patch(circle)
            
            # Ajouter des titres et étiquettes
            ax.set_title(f"Indice {correction_ind}, λ = {wavelength} nm")
            ax.set_xlabel("Position X")
            ax.set_ylabel("Position Y")
            ax.set_aspect('equal')
            
            # Ajouter une barre de couleur
            plt.colorbar(contour, ax=ax, label="Erreur (%)")
    
    # Ajouter un titre global
    fig.suptitle("Erreur (%) sur la longueur d'onde estimée selon la position du faisceau", fontsize=16)
    
    # Sauvegarder la figure
    plt.savefig('position_error_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
# Exemple d'utilisation:
if __name__ == "__main__":
    def analyse_sensibilite():
        # Définir les longueurs d'onde à tester (celles mentionnées dans le code original)
        wavelengths = [450, 976, 1976]
        
        # Définir la plage de niveaux de bruit (de 0% à 20% par incréments de 2%)
        noise_levels = np.arange(0, 0.41, 0.05)
        
        # Tester l'influence du bruit
        results, sensor_variations = test_noise_influence(wavelengths, noise_levels, num_iterations=10)
        
        # Tracer les heatmaps séparément
        plot_separate_heatmaps(results, sensor_variations)
        
        # Tracer l'évolution de l'erreur
        plot_wavelength_errors(results)
    
    def extraction_reponses_laser():
        algo = AlgoWavelength(model_path='model_nn_pytorch_weights.pth')

        bruit = [20.02098894, 17.89683616, 15.70257174, 303.10787518, 2.88497193, 995.92691598, 1127.36721619, 388.34201666]
        algo.calculate_wavelength(np.array(bruit), enable_print=True)
        algo.mise_a_zero()
        
        # Test du modèle avec une longueur d'onde spécifique
        test_wavelength = 450
        # ratios_dict, ratios_list = algo.get_sensor_ratios_for_wavelength(test_wavelength)
        responses_dict, responses_list = algo.get_sensor_response_for_wavelength(test_wavelength, enable_print=True)
        ratios_list = np.array(responses_list)/ np.max(responses_list)
        start_pred = perf_counter()
        predicted_wavelength = algo.calculate_wavelength(np.array([x + y for x, y in zip(responses_list, bruit)]), enable_print=True)
        print(f"Temps de prédiction: {perf_counter() - start_pred:.8f} secondes")
        algo.reset_mise_a_zero()
    
        test_wavelength = 976
        # ratios_dict, ratios_list = algo.get_sensor_ratios_for_wavelength(test_wavelength)
        responses_dict, responses_list = algo.get_sensor_response_for_wavelength(test_wavelength, enable_print=True)
        ratios_list = np.array(responses_list) / np.max(responses_list)
        pred2 = perf_counter()
        predicted_wavelength = algo.calculate_wavelength(np.array(responses_list), enable_print=True)
        print(f"Temps de prédiction: {perf_counter() - pred2:.8f} secondes")
        
        test_wavelength = 1976
        # ratios_dict, ratios_list = algo.get_sensor_ratios_for_wavelength(test_wavelength)
        responses_dict, responses_list = algo.get_sensor_response_for_wavelength(test_wavelength, enable_print=True)
        ratios_list = np.array(responses_list) / np.max(responses_list)
        pred3 = perf_counter()
        predicted_wavelength = algo.calculate_wavelength(np.array(responses_list), enable_print=True)
        print(f"Temps de prédiction: {perf_counter() - pred3:.8f} secondes")
        
    def plot_reponses_et_ratios():
        algo = AlgoWavelength(model_path='model_nn_pytorch_weights.pth')
    
        # Afficher les courbes de réponse spectrale
        algo.plot_spectral_response()
        algo.plot_spectral_ratios()
    
    #plot_reponses_et_ratios()
    extraction_reponses_laser()
    #analyse_sensibilite()
    #map_position_error()



