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



    """ 
    # Fonction généré gaussienne
    laser_origin = {"x":0, "y":0}
    
    # Fonction gaussienne centrée en (x_0, y_0) et normalisée entre 0 et 3000
    def gaussian(x, y, x_0=0, y_0=0):
        return 3000 * np.exp(-0.5 * ((x - x_0)**2 + (y - y_0)**2) / 10**2)
    
    # Affichage de la fonction continue gaussienne originale en 3D
    X, Y = np.meshgrid(np.linspace(-rayon, rayon, 300), np.linspace(-rayon, rayon, 300))
    Z = gaussian(X, Y, x_0=laser_origin['x'], y_0=laser_origin['y'])
    plot_3d(X, Y, Z, title="Fonction gaussienne originale")

    #maximum de la fonction gaussienne originale
    max_x, max_y, max_value = find_max_interpolation(X, Y, Z)
    print("Maximum de la fonction gaussienne originale à (x, y) = ({:.2f}, {:.2f}) avec une valeur de {:.2f}".format(max_x, max_y, max_value))


    # Génération des températures sur la grille et ajout de bruit
    Temp = gaussian(position_xy[:, :, 0], position_xy[:, :, 1], x_0=laser_origin['x'], y_0=laser_origin['y'])
    print(Temp) """


    csv_simulation_1 = "Thermique\SimulationCSV\Offset_1_10W_parsed.csv"

    # Define the grid size and spacing
    position_xy = np.array([
        [[-10.5, 10.5], [-3.5, 10.5], [3.5, 10.5], [10.5, 10.5]],
        [[-10.5, 3.5],  [-3.5, 3.5],  [3.5, 3.5],  [10.5, 3.5]],
        [[-10.5, -3.5], [-3.5, -3.5], [3.5, -3.5], [10.5, -3.5]],
        [[-10.5, -10.5],[-3.5, -10.5],[3.5, -10.5], [10.5, -10.5]]
    ])
    


    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_simulation_1)

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

    


    # Prend la coubre de température dans le dossier /Thermique/Simulation 03-26/Test Lecture CSV.py
    #print("heatsink_temperature=", heatsink_temperature)
    #print("temp_min=" , temp_min)
    noise_level = 1
    Temp = temp_simulation_2 - temp_min + -noise_level+noise_level*np.random.rand(4,4)
    print("Temps", Temp)

    rayon = 30
    
    
    
    # Interpolation/extrapolation dans un cercle de rayon 12.5 mm centré en (0,0)
    X, Y, Z = AlgoPosition.interpolate_circle(Temp, position_xy, radius=rayon, center=(0, 0), resolution=300, rbf_function='gaussian')
    
    print(len(X))
    # Affichage de la surface 
    AlgoPosition.plot_matrix_color(Temp+heatsink_temperature)
    AlgoPosition.plot_interpolation_2d(X, Y, Z+heatsink_temperature, original_points=np.concatenate([position_xy, Temp[:, :, None]], axis=2).reshape(-1, 3))
    AlgoPosition.plot_3d(X, Y, Z+heatsink_temperature, title="Interpolation/extrapolation dans un cercle")
  

    # Recherche du maximum de l'interpolation
    max_x, max_y, max_value = AlgoPosition.find_max_interpolation(X, Y, Z)
    print("Maximum interpolé à (x, y) = ({:.2f}, {:.2f}) avec une valeur de {:.2f}".format(max_x, max_y, max_value))


# -------------------- Fin de l'exemple --------------------