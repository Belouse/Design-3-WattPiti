import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Liste des fichiers CSV
fichiers_csv = [
    "thermistancesCalibrationTest_Plaque_Chauffante.csv",
    "thermistancesCalibrationTest_Plaque_Chauffante_Froid.csv",
    "thermistancesCalibrationTest_Plaque_Chauffante_Froid_2.csv"
]

# Charger et combiner les fichiers CSV
dfs = []
for fichier in fichiers_csv:
    try:
        df_temp = pd.read_csv(fichier)
        print(f"Fichier {fichier} chargé: {df_temp.shape[0]} lignes")
        dfs.append(df_temp)
    except FileNotFoundError:
        print(f"Fichier {fichier} non trouvé, ignoré")

# Concaténer les DataFrames
if len(dfs) > 0:
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nDataFrame combiné: {df.shape[0]} lignes")
    print(f"Plage de température: {df['ref'].min():.2f}°C à {df['ref'].max():.2f}°C")
else:
    raise Exception("Aucun fichier CSV n'a pu être chargé")


# Filtrer les données pour ne garder que celles au-dessus de 31°C
df = df[df['ref'] >= 31.0]
#df = df[df['ref'] <= 50.0]  # Limiter à 50°C pour éviter les valeurs extrêmes
print(f"Après filtrage (>= 31°C): {df.shape[0]} lignes")


# Créer la liste de tous les capteurs
capteurs = [f'therm{i}_{j}' for i in range(1, 5) for j in range(1, 5)]

# Extrapoler les données vers les basses températures (jusqu'à 10°C)
temperature_min_cible = 10.0  # °C
temperature_extrapolation_debut = 42.0  # °C
df_extrapolation = []

# Créer une grille de sous-graphiques (4x4)
fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
fig.suptitle("Relation entre température et valeurs des capteurs avec courbes de régression", fontsize=16)

# Analyser chaque capteur et l'afficher dans la grille
for idx, capteur in enumerate(capteurs):
    # Calculer la position dans la grille
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    # Sélectionner les données pour l'extrapolation
    df_segment = df[(df['ref'] <= temperature_extrapolation_debut)]

    if df_segment.empty:
        print(f"Pas assez de données pour extrapoler {capteur}")
        continue

    # Trier par température pour l'extrapolation
    df_segment = df_segment.sort_values('ref')

    # Calculer la régression linéaire sur ce segment
    X = df_segment['ref'].values.reshape(-1, 1)
    y = df_segment[capteur].values

    model = LinearRegression()
    model.fit(X, y)

    # Générer des points extrapolés
    temp_range = np.arange(df['ref'].min(), temperature_min_cible, -0.5)
    counts_extrapolated = model.predict(temp_range.reshape(-1, 1))

    # Créer un dataframe pour les données extrapolées
    df_extrapolated = pd.DataFrame({
        'ref': temp_range,
        capteur: counts_extrapolated
    })

    # Filtrer uniquement les nouvelles températures (inférieures au minimum actuel)
    df_extrapolated = df_extrapolated[df_extrapolated['ref'] < df['ref'].min()]

    # Ajouter à notre liste d'extrapolation
    if len(df_extrapolation) == 0:
        df_extrapolation = df_extrapolated.copy()
    else:
        # S'assurer que nous avons les bonnes colonnes
        for c in df.columns:
            if c not in df_extrapolation.columns and c != capteur:
                df_extrapolation[c] = np.nan

        # Mettre à jour la colonne du capteur actuel
        df_extrapolation[capteur] = df_extrapolated[capteur].values

    # Visualiser les données et l'extrapolation
    ax.scatter(df[capteur], df['ref'], s=3, alpha=0.5, label='Données originales')
    ax.scatter(df_extrapolated[capteur], df_extrapolated['ref'],
               s=3, color='red', alpha=0.7, label='Extrapolation')

    # Tracer la ligne de régression
    line_x = np.array([temperature_min_cible, temperature_extrapolation_debut])
    line_y = model.predict(line_x.reshape(-1, 1))
    ax.plot(line_y, line_x, 'g-', linewidth=1)

    # Configurer l'axe
    ax.set_title(capteur)
    ax.grid(True, alpha=0.3)
    if row == 3:
        ax.set_xlabel('Température (°C)')
    if col == 0:
        ax.set_ylabel('Counts')
    ax.legend(fontsize='x-small')

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()

# Combiner les données originales avec les données extrapolées
df_combined = pd.concat([df, df_extrapolation], ignore_index=True)
print(f"Données originales: {df.shape[0]} lignes")
print(f"Données extrapolées: {df_extrapolation.shape[0]} lignes")
print(f"Données combinées: {df_combined.shape[0]} lignes")
print(f"Nouvelle plage de température: {df_combined['ref'].min():.2f}°C à {df_combined['ref'].max():.2f}°C")


# Créer un dictionnaire pour stocker les modèles de régression
models = {}
degrees = {}  # Pour stocker le meilleur degré pour chaque capteur
rmse_values = {}  # Pour stocker les erreurs RMSE

# Refaire l'analyse avec les données étendues
degree = 4  # Augmenter le degré pour mieux capturer le comportement

# Créer une figure pour visualiser les résultats avec les données étendues
fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
fig.suptitle("Calibration des capteurs avec données étendues: Counts vers Température", fontsize=16)

for idx, capteur in enumerate(capteurs):
    # Calculer la position dans la grille
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]

    # Préparer les données (counts comme variable indépendante)
    sensor_data = df_combined.groupby("ref")[capteur].mean().reset_index()
    sensor_data = sensor_data.sort_values(by=capteur)
    sensor_data = sensor_data.dropna()  # Éliminer les valeurs NaN potentielles

    X = sensor_data[capteur].values.reshape(-1, 1)  # Counts
    y = sensor_data['ref'].values  # Température


    # Créer les caractéristiques polynomiales
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Ajuster le modèle
    model = LinearRegression()
    model.fit(X_poly, y)

    # Faire des prédictions et calculer l'erreur
    y_pred = model.predict(X_poly)
    rmse = np.sqrt(mean_squared_error(y, y_pred))


    # Stocker le meilleur modèle et son degré
    models[capteur] = model
    degrees[capteur] = degree
    rmse_values[capteur] = rmse

    # Afficher les données brutes
    ax.scatter(X, y, s=3, alpha=0.5, label='Données brutes + extrapolées')

    # Préparer les points pour tracer la courbe de régression
    X_range = np.linspace(0, X.max(), 100).reshape(-1, 1)
    X_range_poly = PolynomialFeatures(degree=degree).fit_transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    # Tracer la courbe de régression
    ax.plot(X_range, y_range_pred, 'r-', linewidth=2,
            label=f'Degré {degree}, RMSE={rmse:.3f}°C')

    # Configurer le graphique
    ax.set_title(f"{capteur}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='x-small')

    if row == 3:
        ax.set_xlabel("Counts")
    if col == 0:
        ax.set_ylabel("Température (°C)")

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()

print("\nÉquations de conversion counts -> température pour chaque capteur:")
print("-" * 70)
print("Format: température = f(counts)")
print("-" * 70)

for capteur in capteurs:
    model = models[capteur]
    degree = degrees[capteur]
    coef = model.coef_
    intercept = model.intercept_

    # Construction de l'équation polynomiale
    equation = f"température = {intercept:.6f}"

    # Vérifier si coef est un array 1D ou 2D
    if len(coef.shape) == 1:
        # Cas régression polynomiale où coef est un array 1D
        for i in range(degree):
            coefficient = coef[i]
            if coefficient >= 0:
                equation += f" + {coefficient}×counts"
            else:
                equation += f" - {abs(coefficient)}×counts"

            if i > 0:  # Commencer les exposants à partir du deuxième terme
                equation += f"^{i + 1}"
    else:
        # Cas où coef est un array 2D (PolynomialFeatures)
        for i in range(1, len(coef[0])):  # Ignorer le terme constant
            coefficient = coef[0][i]
            if coefficient >= 0:
                equation += f" + {coefficient}×counts"
            else:
                equation += f" - {abs(coefficient)}×counts"

            # Pour PolynomialFeatures, les puissances ne suivent pas un ordre linéaire simple
            # On peut utiliser cette approximation, mais ce n'est pas toujours exact
            if i > 1:
                equation += f"^{i}"

    print(f"{capteur}: {equation}")
