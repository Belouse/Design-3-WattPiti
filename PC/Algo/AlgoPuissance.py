import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SensorOutput:
    temperature: np.ndarray

class AlgoPower:
    def __init__(self):
        pass

    def calculatePower(self, data):
        temperatures = data.temperature
        thermistor_temps = temperatures[:-1]  # 16 premières valeurs
        T0 = temperatures[-1]                 # Température de référence (17e élément)

        # Rayon plaque (mm)
        radius = 30

        # Positions radiales : 16 points répartis uniformément + 1 point pour T0
        radial_positions = np.linspace(0, radius, len(thermistor_temps) + 1)

        # Températures mesurées incluant la température de référence
        all_temperatures = np.append(thermistor_temps, T0)

        # Approximation d'une gaussienne
        def gaussian(x, a, b, c):
            return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

        # Paramètres de la gaussienne
        a = np.max(thermistor_temps)  
        b = 0                         
        c = radius / 3                

        # Génération des températures gaussiennes
        gaussian_temps = gaussian(radial_positions, a, b, c)

        # Modèle pour le calcul de la puissance --> trouvé avec curvfit des courbes à Tom (*pas précis*)
        def calculate_laser_power(T_r, r):
            return (T_r - T0) / (2.70 * np.exp(-0.0018 * r**2))

        # Calcul des puissances individuelles
        laser_powers = np.array([calculate_laser_power(T_r, r) 
                                for T_r, r in zip(thermistor_temps, radial_positions[:-1])])

        # Puissance totale du laser
        total_power = np.sum(laser_powers)

        # Graphique pour températures
        plt.figure(figsize=(8, 5))
        plt.plot(radial_positions, all_temperatures, 'o-', label='Températures mesurées')
        plt.plot(radial_positions, gaussian_temps, '--', label='Gaussienne ajustée')
        plt.title('Distribution des températures sur la plaque chauffée')
        plt.xlabel('Position radiale (mm)')
        plt.ylabel('Température (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Graphique pour puissances
        plt.figure(figsize=(8, 5))
        plt.plot(radial_positions[:-1], laser_powers, 'o-', label='Puissance estimée (W)')
        plt.title('Estimation de la puissance du laser en fonction de la position radiale')
        plt.xlabel('Position radiale (mm)')
        plt.ylabel('Puissance (W)')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Puissance totale du laser estimée : {total_power:.2f} W")
        return total_power

# Exemple d'utilisation
sensor_data = SensorOutput(
    temperature=np.array([50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 
                          30, 28, 26, 24, 22, 20, 30])  # Dernier élément = température de référence
)

algo = AlgoPower()
algo.calculatePower(sensor_data)