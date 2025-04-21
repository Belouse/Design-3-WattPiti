import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from SerialListenerClass import SerialListener
from SerialManagerClass import SerialManager

# Définir le nom du port série
portName = "/dev/cu.usbmodem334E355C31332"
print(f"Port name: {portName}")


## Initialisation
serialListener = SerialListener(portName)
# serialManager = SerialManager(maxData=1000)


# Vérifier si le port série est défini
if serialListener is None:
    raise Exception("Serial port not set. Please set the port name first.")


# Initialiser le dictionnaire pour stocker les valeurs de thermistance
values_thermistances = {
    "therm1_1": [],
    "therm1_2": [],
    "therm1_3": [],
    "therm1_4": [],
    "therm2_1": [],
    "therm2_2": [],
    "therm2_3": [],
    "therm2_4": [],
    "therm3_1": [],
    "therm3_2": [],
    "therm3_3": [],
    "therm3_4": [],
    "therm4_1": [],
    "therm4_2": [],
    "therm4_3": [],
    "therm4_4": [],
    "ref": []
}

# Nombre de données à lire (nombre de lignes)
numberOfData = 1
numberOfLoops = 12000
timeStep = 0.1

duree_minutes = 10
duree_secondes = duree_minutes * 60
temps_debut = time.time()
temps_fin = temps_debut + duree_secondes

# ================ Loop de test sur "numberOfLoops" itérations ================
# 1 itération = 1 lecture de 17 lignes de données chaque "timeStep" secondes

with tqdm(total=duree_secondes, desc="Acquisition de données", unit="s") as pbar:
    derniere_maj = temps_debut

    while time.time() < temps_fin:


        # Lire les données brutes du port série
        rawData = serialListener.readData(numberOfData, printExecutionTime=False)
        #print(rawData)


        # Isoler les données thermiques (1 ligne = 1 x 17)
        thermaldata = []
        for dic in rawData:
            thermalList = dic["thermal"]
            thermaldata.append(thermalList)

        thermalMatrix = np.array(thermaldata)

        temps_actuel = time.time()
        temps_ecoule = int(temps_actuel - temps_debut)

        # print(f"Temp référence: {thermalMatrix[0][-1]}°C")
        print(thermalMatrix[0])

        # Save the data in the dictionary
        values_thermistances["therm1_1"].append(thermalMatrix[0][0])
        values_thermistances["therm1_2"].append(thermalMatrix[0][1])
        values_thermistances["therm1_3"].append(thermalMatrix[0][2])
        values_thermistances["therm1_4"].append(thermalMatrix[0][3])
        values_thermistances["therm2_1"].append(thermalMatrix[0][4])
        values_thermistances["therm2_2"].append(thermalMatrix[0][5])
        values_thermistances["therm2_3"].append(thermalMatrix[0][6])
        values_thermistances["therm2_4"].append(thermalMatrix[0][7])
        values_thermistances["therm3_1"].append(thermalMatrix[0][8])
        values_thermistances["therm3_2"].append(thermalMatrix[0][9])
        values_thermistances["therm3_3"].append(thermalMatrix[0][10])
        values_thermistances["therm3_4"].append(thermalMatrix[0][11])
        values_thermistances["therm4_1"].append(thermalMatrix[0][12])
        values_thermistances["therm4_2"].append(thermalMatrix[0][13])
        values_thermistances["therm4_3"].append(thermalMatrix[0][14])
        values_thermistances["therm4_4"].append(thermalMatrix[0][15])
        values_thermistances["ref"].append(thermalMatrix[0][16])

        pbar.update(temps_actuel - derniere_maj)
        derniere_maj = temps_actuel


    #time.sleep(timeStep)


# ======== Save les données en CSV (une colonne pour chaque thermistance) ========
save = True

if save:
    df = pd.DataFrame(values_thermistances)
    df.to_csv("thermistancesCalibrationTest_Plaque_Chauffante_Froid_2.csv", index=False)


# ================ Affichage des données obtenues test calibration ================

# Afficher tous les résultats de thermistance sur le même graph ou juste une
# plot_all_thermistances = False
#
#
# plt.figure(figsize=(10, 6))
# # Si juste une thermistance, choisir le couple
# therm_i = 3
# therm_j = 4
#
# plt.scatter(values_thermistances["ref"], values_thermistances[f"therm{therm_i}_{therm_j}"], s=2, label=f"therm{therm_i}_{therm_j}")
# plt.xlabel("Température de référence (MCP9808) [°C]")
# plt.ylabel("Counts thermistance [counts]")
# plt.title(f"Courbe de calibration de la thermistance {therm_i}_{therm_j}")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()
#
#
#
# plt.figure(figsize=(10, 6))
# # Afficher toutes les thermistances
# for i in range(1, 5):
#     for j in range(1, 5):
#         plt.scatter(values_thermistances["ref"], values_thermistances[f"therm{i}_{j}"], s=2, label=f"therm{i}_{j}")
#
# plt.xlabel("Température de référence (MCP9808) [°C]")
# plt.ylabel("Counts thermistance [counts]")
# plt.title("Courbes de calibration des thermistances")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.show()




