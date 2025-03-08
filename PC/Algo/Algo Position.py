import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def interpolate_circle(matrix, position, radius, center=(0, 0), resolution=300, rbf_function='gaussian'):
    """
    Réalise l'interpolation et extrapolation des données sur une grille couvrant un cercle.
    
    Paramètres:
      - matrix : matrice de valeurs (par exemple, températures) de dimension (n, m)
      - position : matrice de positions associée de dimension (n, m, 2) où chaque position = [x, y]
      - radius : rayon du cercle d'interpolation
      - center : centre du cercle (tuple (x, y)), par défaut (0, 0)
      - resolution : nombre de points pour la grille d'interpolation (définit la finesse)
      - rbf_function : type de fonction de base radiale utilisée ('multiquadric', 'linear', 'cubic', 'gaussian', etc.)
      
    Retourne:
      - X, Y, Z : grilles de coordonnées et valeurs interpolées (avec masquage des points hors du cercle)
    """
    # Extraction des données
    x_list, y_list, z_list = [], [], []
    n, m = matrix.shape
    for i in range(n):
        for j in range(m):
            x_list.append(position[i, j, 0])
            y_list.append(position[i, j, 1])
            z_list.append(matrix[i, j])
    x = np.array(x_list)
    y = np.array(y_list)
    z = np.array(z_list)
    
    # Création de l'interpolateur RBF
    rbf = Rbf(x, y, z, function=rbf_function)
    
    # Définition de la grille (sur un carré couvrant le cercle)
    xi = np.linspace(center[0] - radius, center[0] + radius, resolution)
    yi = np.linspace(center[1] - radius, center[1] + radius, resolution)
    X, Y = np.meshgrid(xi, yi)
    
    # Évaluation de l'interpolateur sur la grille
    Z = rbf(X, Y)
    
    # Masquage des points situés hors du cercle
    mask = (X - center[0])**2 + (Y - center[1])**2 > radius**2
    Z[mask] = np.nan
    
    return X, Y, Z

def plot_interpolation_3d(X, Y, Z, original_points=None, title='Interpolation dans le cercle'):
    """
    Affiche la surface interpolée en 3D.
    
    Paramètres:
      - X, Y, Z : grilles de coordonnées et valeurs interpolées
      - original_points : tableau numpy de forme (n_points, 3) contenant [x, y, z] des données d'origine (optionnel)
      - title : titre du graphique
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='turbo', edgecolor='none')
        
    ax.set_title(title)
    plt.show()

def find_max_interpolation(X, Y, Z):
    """
    Trouve la position (x, y) et la valeur maximale dans la grille interpolée.
    Prend en compte que certains points peuvent être masqués (np.nan).
    
    Retourne:
      - max_x, max_y, max_value
    """
    # On utilise np.nanargmax pour ignorer les nan
    index_max = np.nanargmax(Z)
    max_value = np.nanmax(Z)
    i, j = np.unravel_index(index_max, Z.shape)
    max_x = X[i, j]
    max_y = Y[i, j]
    return max_x, max_y, max_value

# Fonction de plot en matrice de couleur
def plot_matrix_color(matrix):
    plt.matshow(matrix, cmap='turbo')
    plt.colorbar()
    plt.show()

# -------------------- Exemple d'utilisation --------------------

if __name__ == '__main__':
    # Définition d'une grille de positions 4x4 (en mm)
    position_xy = np.array([
        [[-10.5, 10.5], [-3.5, 10.5], [3.5, 10.5], [10.5, 10.5]],
        [[-10.5, 3.5],  [-3.5, 3.5],  [3.5, 3.5],  [10.5, 3.5]],
        [[-10.5, -3.5], [-3.5, -3.5], [3.5, -3.5], [10.5, -3.5]],
        [[-10.5, -10.5],[-3.5, -10.5],[3.5, -10.5], [10.5, -10.5]]
    ])
    
    # Fonction gaussienne centrée en (x_0, y_0) et normalisée entre 0 et 3000
    def gaussian(x, y, x_0=0, y_0=0):
        return 3000 * np.exp(-0.5 * ((x - x_0)**2 + (y - y_0)**2) / 10**2)
    


    # Génération des températures sur la grille et ajout de bruit
    Temp = gaussian(position_xy[:, :, 0], position_xy[:, :, 1], x_0=5, y_0=5)
    noise_level = 50
    Temp = Temp + np.random.normal(0, noise_level, Temp.shape)
    
    
    # Interpolation/extrapolation dans un cercle de rayon 12.5 mm centré en (0,0)
    X, Y, Z = interpolate_circle(Temp, position_xy, radius=25, center=(0, 0), resolution=300)
    
    # Affichage de la surface 
    plot_matrix_color(Temp)
    plot_interpolation_3d(X, Y, Z, title="Interpolation/extrapolation dans un cercle")
    
    # Recherche du maximum de l'interpolation
    max_x, max_y, max_value = find_max_interpolation(X, Y, Z)
    print("Maximum interpolé à (x, y) = ({:.2f}, {:.2f}) avec une valeur de {:.2f}".format(max_x, max_y, max_value))


# -------------------- Fin de l'exemple --------------------