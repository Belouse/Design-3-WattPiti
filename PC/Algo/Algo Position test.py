


# L'algorithme de positionnement doit prendre en entré la matrice de température 4x4 et retourner la position du maximum de température
# La matrice de température est une d'entier 0-4095

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf
import scipy as sp


# Exemple de température en matrice numpy
Temp = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])


# La position des température receuilli en coordonnée x,y en milimètre
position_xy = np.array([[[-10.5, 10.5], [-3.5, 10.5], [3.5, 10.5], [10.5, 10.5]],
                        [[-10.5, 3.5], [-3.5, 3.5], [3.5, 3.5], [10.5, 3.5]],
                        [[-10.5, -3.5], [-3.5, -3.5], [3.5, -3.5], [10.5, -3.5]],
                        [[-10.5, -10.5], [-3.5, -10.5], [3.5, -10.5], [10.5, -10.5]]])

# Exemple de matrice de température de forme gaussienne centré en 0,0 et normalisé entre 0 et 4095
def gaussian(x, y):
    return 4095 * np.exp(-0.5 * (x ** 2 + y ** 2) / 10 ** 2)

# Fonction qui prend en entré une matrice de température et retourne une matrice avec du bruit
def add_noise(matrix, noise_level):
    return matrix + np.random.normal(0, noise_level, matrix.shape)

# Fonction de plot en matrice de couleur
def plot_matrix_color(matrix):
    plt.matshow(matrix, cmap='turbo')
    plt.colorbar()
    plt.show()

# Fonction qui plot les valeurs de la matrice en 3D avec des points et qui prend en entré la matrice de température et leur position
def plot_matrix_3D(matrix, position):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    for i in range(4):
        for j in range(4):
            x.append(position[i][j][0])
            y.append(position[i][j][1])
            z.append(matrix[i][j])
    ax.scatter(x, y, z)
    plt.show()

# Fonction qui prend en entré une matrice de température, la position et qui plot en 3D en interpolant 
def plot_matrix_3D_interpol(matrix, position):
    x = []
    y = []
    z = []
    for i in range(4):
        for j in range(4):
            x.append(position[i][j][0])
            y.append(position[i][j][1])
            z.append(matrix[i][j])
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = sp.interpolate.griddata((x, y), z, (X, Y), method='cubic')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='turbo')
    plt.show() 




# Fonction qui prend en entré une matrice de température, la position et un certain rayon et qui plot en 3D en interpolant les valeurs dans un cercle de rayon donné
def plot_matrix_3D_interpol_circle_extrap(matrix, position, radius, center=(0, 0)):
    # Extraction des coordonnées et valeurs de température
    x, y, z = [], [], []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x.append(position[i, j, 0])
            y.append(position[i, j, 1])
            z.append(matrix[i, j])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    # Création de l'interpolateur RBF
    rbf = Rbf(x, y, z, function='gaussian') 
    # autres fonctions ('multiquadric', 'linear', 'cubic', 'gaussian', ...)
    # Échanger pour faire le mieux fiter avec les simulations thermique

    # Définition d'une grille couvrant le cercle désiré
    xi = np.linspace(center[0] - radius, center[0] + radius, 300)
    yi = np.linspace(center[1] - radius, center[1] + radius, 300)
    X, Y = np.meshgrid(xi, yi)
    
    # Évaluation de l'interpolateur sur la grille (y compris à l'extérieur du carré initial)
    Z = rbf(X, Y)
    
    # Masquage des points en dehors du cercle
    mask = (X - center[0])**2 + (Y - center[1])**2 > radius**2
    Z[mask] = np.nan
    
    # Affichage en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='turbo', edgecolor='none')
    #ax.scatter(x, y, z, color='red', s=50)  # points d'origine
    plt.title(f'Interpolation extrapolée dans un cercle de rayon {radius} mm')
    plt.show()



# Fonction qui trouve le maximum de la température interpollé et extrapollé et retourne la position
def find_max_temp(matrix, position, radius, center=(0, 0)):
    # Extraction des coordonnées et valeurs de température
    x, y, z = [], [], []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x.append(position[i, j, 0])
            y.append(position[i, j, 1])
            z.append(matrix[i, j])
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    # Création de l'interpolateur RBF
    rbf = Rbf(x, y, z, function='gaussian') 
    # autres fonctions ('multiquadric', 'linear', 'cubic', 'gaussian', ...)
    # Échanger pour faire le mieux fiter avec les simulations thermique

    # Définition d'une grille couvrant le cercle désiré
    xi = np.linspace(center[0] - radius, center[0] + radius, 300)
    yi = np.linspace(center[1] - radius, center[1] + radius, 300)
    X, Y = np.meshgrid(xi, yi)
    
    # Évaluation de l'interpolateur sur la grille (y compris à l'extérieur du carré initial)
    Z = rbf(X, Y)
    
    # Masquage des points en dehors du cercle
    mask = (X - center[0])**2 + (Y - center[1])**2 > radius**2
    Z[mask] = np.nan
    
    # Trouver le maximum de la température
    max_temp = np.nanmax(Z)
    max_temp_pos = np.where(Z == max_temp)
    
    return (X[max_temp_pos], Y[max_temp_pos])



Temp_gaussian = (gaussian(position_xy[:, :, 0], position_xy[:, :, 1]))/5
Temp_gaussian = add_noise(Temp_gaussian, 75)

plot_matrix_color(Temp_gaussian)
#plot_matrix_3D(Temp_gaussian, position_xy)
plot_matrix_3D_interpol(Temp_gaussian, position_xy)
plot_matrix_3D_interpol_circle_extrap(Temp_gaussian, position_xy, 25, center=(0, 0))









