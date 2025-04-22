import json
import serial
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from SerialListenerClass import SerialListener
from DataContainerClass import DataContainer
from SerialManagerClass import SerialManager
from AlgorithmManagerClass import AlgorithmManager


# Définir le port série
portName = "/dev/cu.usbmodem334E355C31332"
print(f"Port name: {portName}")

# Initialisation
dataContainer = DataContainer()
algorithmManager = AlgorithmManager(dataContainer)
serialManager = SerialManager(dataContainer, maxData=100)
serialManager.setPortName(portName)
serialListener = SerialListener(portName)


numberOfLoops = 100
numberOfDataPoints = 1

results = {
    "hotSpot": [],
    "tempAilette": [],
    "temps": [],
    "temps total" : [],
    "thermalCounts": [],
    "tempArray": [],
    "Puissance": [],
    "wavelength": []
}


duree_minutes = 1
duree_secondes = duree_minutes * 60
temps_debut = time.time()
temps_fin = temps_debut + duree_secondes
temps_tot = [temps_debut]


with tqdm(total=duree_secondes, desc="Acquisition de données", unit="s") as pbar:
    derniere_maj = temps_debut

    while time.time() < temps_fin:

        rawData = serialListener.readData(numberOfDataPoints, printExecutionTime=False)
        # print(rawData)

        # Isoler les données thermiques (1 ligne = 1 x 17)
        thermaldata = []
        for dic in rawData:
            thermalList = dic["thermal"]
            thermaldata.append(thermalList)

        thermalMatrix = np.array(thermaldata)

        results["thermalCounts"].append(thermalMatrix)
        #print(results["thermalCounts"])

        serialManager.updateDataFromMCU(numberOfDataPoints)

        algorithmManager.calculatePosition()
        algorithmManager.calculatePower()
        hotSpot = dataContainer.max_temperature
        tempAilette = dataContainer.temperature[-1]
        dt = dataContainer.Delta_t
        tempArray = dataContainer.temperature
        puissance = dataContainer.power

        algorithmManager.calculateWavelength()
        wavelength = dataContainer.wavelength

        results["temps total"].append(time.time())
        results["hotSpot"].append(hotSpot)
        results["tempAilette"].append(tempAilette)
        results["temps"].append(dt)
        results["tempArray"].append(tempArray)
        results["Puissance"].append(puissance)
        results["wavelength"].append(wavelength)

        temps_tot.append(time.time()-temps_debut)

        print(f"Hot Spot: {hotSpot} °C")
        print(f"Temp Ailette: {tempAilette} °C")
        print(f"Longueur d'onde: {dataContainer.wavelength} nm")
        print(f"Puissance: {puissance} W")
        print("\n")

        temps_actuel = time.time()
        temps_ecoule = int(temps_actuel - temps_debut)
        pbar.update(temps_actuel - derniere_maj)
        derniere_maj = temps_actuel


# ======== Save les données en CSV (une colonne pour chaque thermistance) ========
save = True

if save:
    df = pd.DataFrame(results)
    df.to_csv("calibPuissance_450_1min_5W.csv", index=False)


# ======== Plot les données ========
# plt.figure(figsize=(10, 5))
# plt.plot(temps_tot, results["hotSpot"], label="Hot Spot", color="red")
# plt.plot(temps_tot, results["tempAilette"], label="Temp Ailette", color="blue")
# plt.xlabel("Temps (s)")
# plt.ylabel("Température (°C)")
# plt.title("Température du Hot Spot et de l'Ailette")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

