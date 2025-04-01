import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AlgoPosition import AlgoPosition
from DataContainerClass import DataContainer

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import pandas as pd


TestPosition = AlgoPosition()




# -------------------- Exemple d'utilisation --------------------
if __name__ == '__main__':
    """
    Exemple d'utilisation de l'interpolation/extrapolation dans un cercle.

    En premier, montre la fonction originale gaussienne en 3D.
    Ensuite, génère des températures sur une grille 4x4 avec ajout de bruit.
    Réalise l'interpolation/extrapolation dans un cercle de rayon 12.5 mm.
    Plot les résultats en 2D et 3D.
    """


    csv_simulation_1 = "Thermique\SimulationCSV\Offset_1_10W_parsed.csv"
    csv_simulation_2 = "Thermique\SimulationCSV\TestEchelon75W.csv"

    # Define the grid size and spacing
    position_xy = np.array([
        [[-10.5, 10.5], [-3.5, 10.5], [3.5, 10.5], [10.5, 10.5]],
        [[-10.5, 3.5],  [-3.5, 3.5],  [3.5, 3.5],  [10.5, 3.5]],
        [[-10.5, -3.5], [-3.5, -3.5], [3.5, -3.5], [10.5, -3.5]],
        [[-10.5, -10.5],[-3.5, -10.5],[3.5, -10.5], [10.5, -10.5]]
    ])
    


    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_simulation_2)

    # create de X, Y arrays with the themperature values
    x_simulation_1 = df['COORDINATES.X']
    y_simulation_1 = df['COORDINATES.Y']
    temp_simulation_1 = df['NDTEMP.T']

    heatsink_temperature = 39

    #plot the data in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_simulation_1, y_simulation_1, temp_simulation_1, c=temp_simulation_1, cmap='turbo')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature (°C)')
    fig.suptitle('Temperature distribution')
    plt.show()
    

    # transform the temp <class 'pandas.core.series.Series'> to a numpy array
    x_simulation_1 = x_simulation_1.to_numpy()
    y_simulation_1 = y_simulation_1.to_numpy()
    temp_simulation_1 = temp_simulation_1.to_numpy()

    #print(temp_simulation_1)

    # find the maximum of the temperature and print the position
    temp_max = np.max(temp_simulation_1)
    indexe_max = np.unravel_index(np.argmax(temp_simulation_1), temp_simulation_1.shape)
    print(max(temp_simulation_1))
    print(indexe_max[0])
    print("position max data =", ((float(x_simulation_1[indexe_max[0]]), float(y_simulation_1[indexe_max[0]])) )) 
    
    # find the minimum of the temperature 
    temp_min = np.min(temp_simulation_1)


    # loop throuhgt the x and y arrays and return the 16 temprature where the position match the position_xy
    # create a 4x4 matrix with the temperature values
    temp_simulation_2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            # find the row where the x and y are the CLOSEST the position_xy
            index = np.argmin(np.abs(x_simulation_1 - position_xy[i][j][0]) + np.abs(y_simulation_1 - position_xy[i][j][1]))
            #print("index", index, "position xy", position_xy[i][j], "x", x_simulation_1[index], "y", y_simulation_1[index])
            # get the temperature value at the index
            temp_simulation_2[i][j] = temp_simulation_1[index]


    # print the temperature matrix
    #print("Température à chaque position :")
    #print(position_xy)
    #print("Température (°C) :")
    #print(temp_simulation_2)

    temp_simulation_3 = np.hstack(temp_simulation_2)
    temp_simulation_3 = np.append(temp_simulation_3, 0)
    #print("temp_simulation_3= ",temp_simulation_3)


    data = DataContainer()
    AlgoPositionInstance = AlgoPosition()
    data.temperature = np.array([39.4394989,  40.51720047, 40.40250015, 39.22230148, 41.09389877, 43.10359955, 42.76750183, 40.5965004,43.02610016, 48.70330048, 47.12360001, 41.83250046,42.22079849, 58.96350098, 50.47050095, 40.92689896, 0])
    print("data before =", data)
    position_calculé = AlgoPosition.calculatePosition(AlgoPositionInstance,data)
    print("Position calcul=", position_calculé)
    print("data after =", data)

    print("data.thermalCaptorPosition =", data.thermalCaptorPosition)
    
    # Plot in 3D, take the X,Y,Z value
    AlgoPosition.plot_2d_v2(data.interpolatedTemperatureGrid[0], data.interpolatedTemperatureGrid[1], data.interpolatedTemperatureGrid[2], original_points=data.thermalCaptorPosition)
    

    if(True):
        # Prend la coubre de température dans le dossier /Thermique/Simulation 03-26/Test Lecture CSV.py
        #print("heatsink_temperature=", heatsink_temperature)
        #print("temp_min=" , temp_min)
        noise_level = 1
        Temp = temp_simulation_2 - temp_min + -noise_level+noise_level*np.random.rand(4,4)
        print("Temps", Temp)

        rayon = 30
        
        
        
        # Interpolation/extrapolation dans un cercle de rayon 12.5 mm centré en (0,0)
        X, Y, Z = AlgoPosition.interpolate_circle(Temp, position_xy, radius=rayon, center=(0, 0), resolution=300, rbf_function='gaussian')

        # Affichage de la surface 
        AlgoPosition.plot_matrix_color(Temp+heatsink_temperature)

        print("Temp=", Temp)
        
        AlgoPosition.plot_2d(X, Y, Z+heatsink_temperature, original_points=np.concatenate([position_xy, Temp[:, :, None]], axis=2).reshape(-1, 3))
        AlgoPosition.plot_3d(X, Y, Z+heatsink_temperature, title="Interpolation/extrapolation dans un cercle")
    

        # Recherche du maximum de l'interpolation
        max_x, max_y, max_value = AlgoPosition.find_max_interpolation(X, Y, Z)
        print("Maximum interpolé à (x, y) = ({:.2f}, {:.2f}) avec une valeur de {:.2f}".format(max_x, max_y, max_value))


    # -------------------- Fin de l'exemple --------------------