import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from DataContainerClass import DataContainer
import pandas as pd
from scipy.interpolate import interp1d
import os

class AlgoPower:
    def __init__(self):
        pass

    def calculatePower(self, dataContainer):
        """
        dataContainer: DataContainer object (see class declaration for details)
        
        algo qui sort une puissance ici... pas de graphiques ici svp, cette fonction là est appelée dans le script main pour tester
        l'interface, donc on veut bien avoir une valeur ici sans interrompre le code:)
        """

        power = 5

        return power



    #def calculatePowerTest(self, nom_fichier):
        
        # Aller dans /Thermique/SimulationCSV depuis le script dans PC/
        #base_path = os.path.dirname(os.path.abspath(__file__))  # /Design-3-WattPiti/PC
        #dossier_csv = os.path.join(base_path, "..", "Thermique", "SimulationCSV")
        #dossier_csv = os.path.normpath(dossier_csv)  # Nettoyer le chemin

        # Créer le chemin complet vers le fichier
        #chemin_fichier = os.path.join(dossier_csv, nom_fichier)

        # --- Données réponses aux échelons pour interpolation ---
        #T_points = np.array([3.73, 10.53, 17.33, 24.13])
        #K_points = np.array([1.492, 2.106, 2.311, 2.413])
        #tau_points = np.array([1.0196, 1.3203, 1.3881, 1.4183])

        # Fonctions interpolées linéaires
        #calculate_K = interp1d(T_points, K_points, kind='linear', fill_value="extrapolate")
        #calculate_tau = interp1d(T_points, tau_points, kind='linear', fill_value="extrapolate")


        # Lire le fichier CSV
        #df = pd.read_csv(chemin_fichier)
    
        # Garder uniquement les données des nodes 121 et 241
        #df_121 = df[df["Node_ID"] == 121]
        #df_241 = df[df["Node_ID"] == 241]

        # S'assurer qu'ils sont bien triés par le temps
        #df_121 = df_121.sort_values(by="Time")
        #df_241 = df_241.sort_values(by="Time")
    
        # Extraire le temps et les températures
        #temps = df_121["Time"].values
        #T_121 = df_121["NDTEMP.T"].values # max_temperature in dataContainerClass
        #T_241 = df_241["NDTEMP.T"].values # 17e élément du vecteur temperature in dataContainerClass
    
        # Calcul de la température normalisée
        #T_norm = T_121 - T_241

        # Supprimer les premières valeurs (trimer pour avoir la bonne valeur de température)
        #T_norm = T_norm[8:] - T_norm[8]
        #temps = temps[8:] - temps[8]

        # Identifier l’indice correspondant à t >= 10 secondes
        #indices_regime_permanent = np.where(temps >= 10)[0]

        # Prendre la première valeur après 10 secondes
        #i = indices_regime_permanent[0]
        #T_regime = T_norm[i]
        #K_regime = calculate_K(T_regime)
        #P_regime = T_regime / K_regime

        # Calcul du terme transitoire de puissance
        #tau_regime = calculate_tau(T_regime)
        #dT_dt = np.gradient(T_norm, temps)
        #dT_dt_regime = dT_dt[i]
        #P_transitoire = (tau_regime / K_regime) * dT_dt_regime

        #Puissance totale
        #P_total = P_regime + P_transitoire

        # Affichage
        #print(f"--- Terme transitoire ---")
        #print(f"Constante de temps tau(T) = {tau_regime:.5f}")
        #print(f"dT/dt ≈ {dT_dt_regime:.5f}")
        #print(f"Terme transitoire (tau/K * dT/dt) = {P_transitoire:.5f}")

        #print(f"\n--- Puissance totale ---")
        #print(f"P_total = P_regime + P_transitoire = {P_total:.5f}")


        #print(f"--- Régime permanent à t = {temps[i]:.2f} s ---")
        #print(f"Température normalisée T(t) = {T_regime:.5f}")
        #print(f"Gain K(T) = {K_regime:.5f}")
        #print(f"Puissance P(t) = T(t) / K(T) = {P_regime:.5f}")
    
        #return P_total

# Crée une instance de la classe
#mon_algo = AlgoPower()
#mon_algo.calculatePowerTest("TestEchelon5W.csv") # mettre le nom du fichier test


######################

    def calculer_puissance(self, container):
        temperature_ailette = container.temperature[-1]
        T_t = container.max_temperature - temperature_ailette

        dT_dt = ((container.max_temperature - temperature_ailette) -(container.old_max_temperature - temperature_ailette)) / container.Delta_t

        K = 2.704
        tau = 0.4553

        P = T_t / K + (tau / K) * dT_dt

        #print(f"T(t) = {T_t:.5f}")
        #print(f"dT/dt = {dT_dt:.5f}")
        #print(f"K(T) = {K:.5f}")
        #print(f"tau(T) = {tau:.5f}")
        #print(f"Puissance P(t) = {P:.5f} W")

        return P

        
        
if __name__ == "__main__":
    # redéfinir une classe simul
    class FakeDataContainer:
        def __init__(self):
            self.max_temperature = 32.35
            self.old_max_temperature = 32.22
            self.temperature = [0]*16 + [25.4]  # ailette
            self.Delta_t =  17.2 - 11.43

    # Créer l'objet test
    container = FakeDataContainer()
    algo = AlgoPower()

    # Lancer le test
    puissance = algo.calculer_puissance(container)

    print(f"\n Puissance calculée : {puissance:.5f} W")