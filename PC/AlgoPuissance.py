import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from DataContainerClass import DataContainer
import pandas as pd
from scipy.interpolate import interp1d

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



    def calculatePowerTest(self, fichier_csv):
        """
        
        Fonction de test ici qui n'interrompt pas le main...
        
        """
        # --- Données réponses aux échelons pour interpolation ---
        T_points = np.array([3.73, 10.53, 17.33, 24.13])
        K_points = np.array([1.492, 2.106, 2.311, 2.413])
        tau_points = np.array([1.0196, 1.3203, 1.3881, 1.4183])

        # Fonctions interpolées linéaires
        calculate_K = interp1d(T_points, K_points, kind='linear', fill_value="extrapolate")
        calculate_tau = interp1d(T_points, tau_points, kind='linear', fill_value="extrapolate")


        # Lire le fichier CSV
        df = pd.read_csv(fichier_csv)
    
        # Garder uniquement les données des nodes 121 et 241
        df_121 = df[df["Node_ID"] == 121]
        df_241 = df[df["Node_ID"] == 241]

        # S'assurer qu'ils sont bien triés par le temps
        df_121 = df_121.sort_values(by="Time")
        df_241 = df_241.sort_values(by="Time")
    
        # Extraire le temps et les températures
        temps = df_121["Time"].values
        T_121 = df_121["NDTEMP.T"].values # max_temperature in dataContainerClass
        T_241 = df_241["NDTEMP.T"].values # 17e élément du vecteur temperature in dataContainerClass
    
        # Calcul de la température normalisée
        T_norm = T_121 - T_241

        # Supprimer les premières valeurs (trimer pour avoir la bonne valeur de température)
        T_norm = T_norm[8:] - T_norm[8]
        temps = temps[8:] - temps[8]

        # Identifier l’indice correspondant à t >= 10 secondes
        indices_regime_permanent = np.where(temps >= 10)[0]

        # Prendre la première valeur après 10 secondes
        i = indices_regime_permanent[0]
        T_regime = T_norm[i]
        K_regime = calculate_K(T_regime)
        P_regime = T_regime / K_regime

        # Calcul du terme transitoire de puissance
        tau_regime = calculate_tau(T_regime)
        dT_dt = np.gradient(T_norm, temps)
        dT_dt_regime = dT_dt[i]
        P_transitoire = (tau_regime / K_regime) * dT_dt_regime

        #Puissance totale
        P_total = P_regime + P_transitoire

        # Affichage
        print(f"--- Terme transitoire ---")
        print(f"Constante de temps tau(T) = {tau_regime:.5f}")
        print(f"dT/dt ≈ {dT_dt_regime:.5f}")
        print(f"Terme transitoire (tau/K * dT/dt) = {P_transitoire:.5f}")

        print(f"\n--- Puissance totale ---")
        print(f"P_total = P_regime + P_transitoire = {P_total:.5f}")


        print(f"--- Régime permanent à t = {temps[i]:.2f} s ---")
        print(f"Température normalisée T(t) = {T_regime:.5f}")
        print(f"Gain K(T) = {K_regime:.5f}")
        print(f"Puissance P(t) = T(t) / K(T) = {P_regime:.5f}")
    
        return P_total

# Crée une instance de la classe
mon_algo = AlgoPower()

# Chemin vers ton fichier CSV
fichier = "/Users/vincentlelievre/Desktop/Design_3/Fichiers_test/TestEchelon10W.csv"

# Appel de la méthode
puissance = mon_algo.calculatePowerTest(fichier)
