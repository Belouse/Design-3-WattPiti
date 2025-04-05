import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import constants 
import pickle
import os
from EntrainementLambda import EntrainementLambda
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm



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
            with open(model_path_abs, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")
        
        # Stocker l'ordre des capteurs pour référence
        self.sensor_order = ['P_IR1', 'P_IR1xP', 'P_IR2', 'P_UV', 'C_UV', 'C_VISG', 'C_VISB', 'C_VISR']
        
        # Créer une instance de EntrainementLambda pour avoir accès aux réponses des capteurs
        self.entrainement = EntrainementLambda()
        self.responses = self.entrainement.all_sensors
            
    def calculate_wavelength(self, sensor_values):
        """
        Prédit la longueur d'onde à partir des valeurs des capteurs.
        
        Parameters:
        sensor_values (list or numpy.ndarray): Liste ou tableau des valeurs normalisées des capteurs
                                               dans l'ordre [P_IR1, P_IR1xP, P_IR2, P_UV, C_UV, C_VISG, C_VISB, C_VISR]
        
        Returns:
        float: Longueur d'onde prédite en nanomètres
        """
        # Convertir les valeurs d'entrée en tableau numpy et les reformater pour la prédiction
        if isinstance(sensor_values, list):
            sensor_values = np.array(sensor_values)
        
        # Reformater pour avoir la forme attendue par le modèle (1 échantillon, 8 caractéristiques)
        if sensor_values.ndim == 1:
            sensor_values = sensor_values.reshape(1, -1)
        
        # Faire la prédiction
        predicted_wavelength = self.model.predict(sensor_values)[0]
        
        return predicted_wavelength
        
    def get_sensor_ratios_for_wavelength(self, wavelength, entrainement_instance=None):
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
        # Utiliser l'instance interne si aucune n'est fournie
        if entrainement_instance is None:
            sensors_data = self.responses
        else:
            sensors_data = entrainement_instance.all_sensors
        
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
        
        return ratios_dict, ratios_list

    def test_model_with_wavelength(self, test_wavelength, entrainement_instance=None):
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
        
        # print(f"Longueur d'onde de test: {test_wavelength} nm")
        # print(f"Longueur d'onde prédite: {predicted_wavelength:.2f} nm")
        # print(f"Erreur absolue: {error:.2f} nm")
        
        return test_wavelength, predicted_wavelength, error
    
    def plot_spectral_response(self):
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

    def sweep_wavelength_range(self, start_nm=300, end_nm=2000, step_nm=50):
        """
        Teste le modèle sur une plage de longueurs d'onde et affiche les résultats.
        
        Parameters:
        start_nm (float): Longueur d'onde de départ (en nm)
        end_nm (float): Longueur d'onde de fin (en nm)
        step_nm (float): Pas entre les longueurs d'onde (en nm)
        
        Returns:
        tuple: (tableau des longueurs d'onde de test, tableau des longueurs d'onde prédites, tableau des erreurs)
        """
        # Créer la plage de longueurs d'onde à tester
        test_wavelengths = np.arange(start_nm, end_nm + step_nm, step_nm)
        
        # Tableaux pour stocker les résultats
        predicted_wavelengths = []
        errors = []
        
        # Tester chaque longueur d'onde
        for wavelength in test_wavelengths:
            _, predicted, error = self.test_model_with_wavelength(wavelength)
            predicted_wavelengths.append(predicted)
            errors.append(error)
        
        # Convertir en tableaux numpy
        predicted_wavelengths = np.array(predicted_wavelengths)
        errors = np.array(errors)
        
        # Afficher les résultats
        plt.figure(figsize=(12, 8))
        
        # Sous-graphique 1: Prédictions vs réalité
        plt.subplot(2, 1, 1)
        plt.plot(test_wavelengths, predicted_wavelengths, 'o-', label="Prédictions")
        plt.plot(test_wavelengths, test_wavelengths, 'r--', label="Référence (y=x)")
        plt.title("Prédictions vs Longueurs d'onde réelles")
        plt.xlabel("Longueur d'onde réelle (nm)")
        plt.ylabel("Longueur d'onde prédite (nm)")
        plt.legend()
        plt.grid(True)
        
        # Sous-graphique 2: Erreurs
        plt.subplot(2, 1, 2)
        plt.bar(test_wavelengths, errors)
        plt.title("Erreur absolue par longueur d'onde")
        plt.xlabel("Longueur d'onde (nm)")
        plt.ylabel("Erreur absolue (nm)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Statistiques globales
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        # print(f"Erreur moyenne: {mean_error:.2f} nm")
        # print(f"Erreur maximale: {max_error:.2f} nm")
        
        return test_wavelengths, predicted_wavelengths, errors
    
    def add_noise_to_ratios(self, ratios, noise_level):
        """
        Ajoute du bruit gaussien aux ratios des capteurs.
        
        Parameters:
        ratios (list or numpy.ndarray): Ratios des capteurs
        noise_level (float): Niveau de bruit (écart-type du bruit gaussien)
        
        Returns:
        numpy.ndarray: Ratios avec bruit ajouté
        """
        ratios_array = np.array(ratios)
        noise = np.random.normal(0, noise_level, size=ratios_array.shape)
        noisy_ratios = ratios_array + noise
        
        # S'assurer que les ratios restent dans l'intervalle [0, 1]
        noisy_ratios = np.clip(noisy_ratios, 0, 1)
        
        return noisy_ratios

    def analyze_noise_impact(self, wavelength_range=(300, 2000), num_wavelengths=10, 
                            noise_levels=(0, 0.01, 0.02, 0.05, 0.1, 0.2), trials_per_level=30):
        """
        Analyse l'impact du bruit sur la précision de l'estimation de la longueur d'onde.
        
        Parameters:
        wavelength_range (tuple): Plage de longueurs d'onde à tester (min, max)
        num_wavelengths (int): Nombre de longueurs d'onde à tester dans la plage
        noise_levels (tuple): Niveaux de bruit à tester
        trials_per_level (int): Nombre d'essais par niveau de bruit
        
        Returns:
        pandas.DataFrame: Dataframe contenant les résultats de l'analyse
        """
        # Longueurs d'onde à tester
        test_wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)
        
        # Préparer le DataFrame pour collecter les résultats
        results = []
        
        # Boucle principale pour chaque longueur d'onde et niveau de bruit
        for wavelength in tqdm(test_wavelengths, desc="Wavelengths"):
            # Obtenir les ratios de référence pour cette longueur d'onde
            _, clean_ratios = self.get_sensor_ratios_for_wavelength(wavelength)
            
            for noise_level in noise_levels:
                for trial in range(trials_per_level):
                    # Ajouter du bruit aux ratios
                    noisy_ratios = self.add_noise_to_ratios(clean_ratios, noise_level)
                    
                    # Prédire la longueur d'onde avec les ratios bruités
                    predicted_wavelength = self.calculate_wavelength(noisy_ratios)
                    
                    # Calculer l'erreur
                    error = abs(predicted_wavelength - wavelength)
                    relative_error = error / wavelength * 100  # en pourcentage
                    
                    # Stocker les résultats
                    results.append({
                        'True Wavelength': wavelength,
                        'Predicted Wavelength': predicted_wavelength,
                        'Noise Level': noise_level,
                        'Absolute Error': error,
                        'Relative Error (%)': relative_error,
                        'Trial': trial,
                    })
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        
        # Afficher les résultats
        self.plot_noise_analysis_results(results_df)
        
        return results_df


    def plot_noise_analysis_results(self, results_df):
        """
        Trace les résultats de l'analyse de bruit.
        
        Parameters:
        results_df (pandas.DataFrame): DataFrame contenant les résultats de l'analyse
        """
        # Créer une figure avec plusieurs sous-graphiques
        plt.figure(figsize=(15, 15))
        gs = GridSpec(3, 2, figure=plt.gcf())
        
        # 1. Erreur absolue moyenne vs Niveau de bruit
        ax1 = plt.subplot(gs[0, 0])
        error_by_noise = results_df.groupby('Noise Level')['Absolute Error'].mean().reset_index()
        ax1.plot(error_by_noise['Noise Level'], error_by_noise['Absolute Error'], 'o-', linewidth=2)
        ax1.set_xlabel('Niveau de bruit')
        ax1.set_ylabel('Erreur absolue moyenne (nm)')
        ax1.set_title('Impact du niveau de bruit sur l\'erreur absolue moyenne')
        ax1.grid(True)
        
        # 2. Box plot de l'erreur par niveau de bruit
        ax2 = plt.subplot(gs[0, 1])
        sns.boxplot(x='Noise Level', y='Absolute Error', data=results_df, ax=ax2)
        ax2.set_xlabel('Niveau de bruit')
        ax2.set_ylabel('Erreur absolue (nm)')
        ax2.set_title('Distribution des erreurs par niveau de bruit')
        
        # 3. Erreur relative (%) par longueur d'onde et niveau de bruit (heatmap)
        ax3 = plt.subplot(gs[1, :])
        heatmap_data = results_df.groupby(['True Wavelength', 'Noise Level'])['Relative Error (%)'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax3)
        ax3.set_xlabel('Niveau de bruit')
        ax3.set_ylabel('Longueur d\'onde réelle (nm)')
        ax3.set_title('Erreur relative moyenne (%) par longueur d\'onde et niveau de bruit')
        
        # 4. Scatter plot de longueur d'onde prédite vs longueur d'onde réelle pour différents niveaux de bruit
        ax4 = plt.subplot(gs[2, :])
        
        # Sélectionner un sous-ensemble pour rendre le graphique plus lisible
        # Prendre uniquement le premier essai pour chaque combinaison
        subset = results_df[results_df['Trial'] == 0]
        
        noise_colors = {
            0: 'green',
            0.01: 'blue',
            0.02: 'purple',
            0.05: 'orange',
            0.1: 'red',
            0.2: 'black'
        }
        
        for noise_level, color in noise_colors.items():
            if noise_level in subset['Noise Level'].values:
                data = subset[subset['Noise Level'] == noise_level]
                ax4.scatter(data['True Wavelength'], data['Predicted Wavelength'], 
                        label=f'Bruit {noise_level}', color=color, alpha=0.7)
        
        # Ligne de référence y=x
        min_wl = results_df['True Wavelength'].min()
        max_wl = results_df['True Wavelength'].max()
        ax4.plot([min_wl, max_wl], [min_wl, max_wl], 'k--', label='Idéal (y=x)')
        
        ax4.set_xlabel('Longueur d\'onde réelle (nm)')
        ax4.set_ylabel('Longueur d\'onde prédite (nm)')
        ax4.set_title('Prédictions vs Réalité pour différents niveaux de bruit')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig("analyse_bruit_longueur_onde.png", dpi=300)
        plt.show()


    def analyze_sensor_sensitivity(self, wavelength_range=(300, 2000), num_wavelengths=10, noise_level=0.05, trials=30):
        """
        Analyse la sensibilité de chaque capteur au bruit pour différentes longueurs d'onde.
        
        Parameters:
        wavelength_range (tuple): Plage de longueurs d'onde à tester (min, max)
        num_wavelengths (int): Nombre de longueurs d'onde à tester dans la plage
        noise_level (float): Niveau de bruit à appliquer
        trials (int): Nombre d'essais pour chaque configuration
        
        Returns:
        pandas.DataFrame: Dataframe contenant les résultats de l'analyse
        """
        # Longueurs d'onde à tester
        test_wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)
        
        # Préparer le DataFrame pour collecter les résultats
        results = []
        
        # Boucle principale pour chaque longueur d'onde
        for wavelength in tqdm(test_wavelengths, desc="Wavelengths"):
            # Obtenir les ratios de référence pour cette longueur d'onde
            _, clean_ratios = self.get_sensor_ratios_for_wavelength(wavelength)
            clean_ratios = np.array(clean_ratios)
            
            # Référence: prédiction sans bruit
            reference_prediction = self.calculate_wavelength(clean_ratios)
            
            # Tester chaque capteur individuellement
            for i, sensor_name in enumerate(self.sensor_order):
                for trial in range(trials):
                    # Créer un ensemble de ratios où seul le capteur actuel a du bruit
                    noisy_ratios = clean_ratios.copy()
                    noisy_ratios[i] += np.random.normal(0, noise_level)
                    noisy_ratios = np.clip(noisy_ratios, 0, 1)  # Limiter entre 0 et 1
                    
                    # Prédire la longueur d'onde avec ce capteur bruité
                    predicted_wavelength = self.calculate_wavelength(noisy_ratios)
                    
                    # Calculer l'erreur
                    error = abs(predicted_wavelength - wavelength)
                    
                    # Stocker les résultats
                    results.append({
                        'True Wavelength': wavelength,
                        'Predicted Wavelength': predicted_wavelength,
                        'Sensor': sensor_name,
                        'Absolute Error': error,
                        'Relative Error (%)': error / wavelength * 100,
                        'Trial': trial,
                    })
        
        # Convertir en DataFrame
        results_df = pd.DataFrame(results)
        
        # Afficher les résultats
        self.plot_sensor_sensitivity_results(results_df)
        
        return results_df


    def plot_sensor_sensitivity_results(self, results_df):
        """
        Trace les résultats de l'analyse de sensibilité des capteurs.
        
        Parameters:
        results_df (pandas.DataFrame): DataFrame contenant les résultats de l'analyse
        """
        plt.figure(figsize=(15, 10))
        
        # 1. Erreur moyenne par capteur
        plt.subplot(2, 2, 1)
        error_by_sensor = results_df.groupby('Sensor')['Absolute Error'].mean().reset_index()
        error_by_sensor = error_by_sensor.sort_values('Absolute Error', ascending=False)
        
        bars = plt.bar(error_by_sensor['Sensor'], error_by_sensor['Absolute Error'])
        plt.xlabel('Capteur')
        plt.ylabel('Erreur absolue moyenne (nm)')
        plt.title('Sensibilité des capteurs au bruit')
        plt.xticks(rotation=45)
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 2. Box plot de l'erreur par capteur
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Sensor', y='Absolute Error', data=results_df)
        plt.xlabel('Capteur')
        plt.ylabel('Erreur absolue (nm)')
        plt.title('Distribution des erreurs par capteur')
        plt.xticks(rotation=45)
        
        # 3. Heatmap de l'erreur relative par capteur et longueur d'onde
        plt.subplot(2, 1, 2)
        heatmap_data = results_df.groupby(['True Wavelength', 'Sensor'])['Relative Error (%)'].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
        plt.xlabel('Capteur')
        plt.ylabel('Longueur d\'onde réelle (nm)')
        plt.title('Erreur relative moyenne (%) par longueur d\'onde et capteur')
        
        plt.tight_layout()
        plt.savefig("analyse_sensibilite_capteurs.png", dpi=300)
        plt.show()

# Exemple d'utilisation des nouvelles fonctions
if __name__ == "__main__":  
    # Création d'une instance de AlgoWavelength
    algo = AlgoWavelength(model_path='heavyModel.pkl')
    
    # Afficher les courbes de réponse spectrale
    algo.plot_spectral_response()
    
    
    
    # Test du modèle avec une longueur d'onde spécifique
    test_wavelength = 800  # Par exemple, 800 nm
    print(algo.test_model_with_wavelength(test_wavelength))
    
    # # Tester plusieurs longueurs d'onde
    # algo.sweep_wavelength_range(start_nm=250, end_nm=2500, step_nm=1)
    
    # # Obtenir les ratios des capteurs pour une longueur d'onde spécifique
    # ratios_dict, ratios_list = algo.get_sensor_ratios_for_wavelength(test_wavelength)

    # print("\nRatios des capteurs pour {} nm:".format(test_wavelength))
    # for sensor, ratio in ratios_dict.items():
    #     print(f"{sensor}: {ratio:.4f}")
    
    # Simuler un cas d'utilisation réel où on aurait des valeurs de capteurs
    # et on voudrait prédire la longueur d'onde
    
    # print("\nTest avec les ratios extraits:")
    # predicted_wavelength = algo.calculate_wavelength(ratios_list)
    # print(f"Longueur d'onde prédite: {predicted_wavelength:.2f} nm")
    
    def analyse_bruit():
        # 1. Analyse de l'impact du bruit sur différentes longueurs d'onde
        print("Analyse de l'impact du bruit...")
        results = algo.analyze_noise_impact(
            wavelength_range=(250, 2500),  # Plage de longueurs d'onde (nm)
            num_wavelengths=10,            # Nombre de longueurs d'onde à tester
            noise_levels=(0, 0.01, 0.02, 0.05, 0.1, 0.2),  # Différents niveaux de bruit
            trials_per_level=20            # Nombre d'essais par configuration
        )
        
        # Sauvegarder les résultats dans un fichier CSV
        results.to_csv("resultats_analyse_bruit.csv", index=False)
        print("Analyse terminée et résultats sauvegardés.")
    
    def analyse_sensibilite_capteurs():
        # 2. Analyse de la sensibilité de chaque capteur
        print("Analyse de la sensibilité des capteurs...")
        sensor_results = algo.analyze_sensor_sensitivity(
            wavelength_range=(250, 2500),  # Plage de longueurs d'onde (nm)
            num_wavelengths=10,             # Nombre de longueurs d'onde à tester
            noise_level=0.05,              # Niveau de bruit à appliquer
            trials=15                      # Nombre d'essais par configuration
        )
        
        # Sauvegarder les résultats dans un fichier CSV
        sensor_results.to_csv("resultats_sensibilite_capteurs.csv", index=False)
        print("Analyse terminée et résultats sauvegardés.")
    
    