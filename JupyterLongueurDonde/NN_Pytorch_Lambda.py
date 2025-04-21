import os
import numpy as np
import pandas as pd
from scipy import constants
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from CapteursDataProcess import DataPreProcess

# -------- Organisation des données dans une classe Dataset PyTorch -------
class CapteursDataset(Dataset):
    """
    Dataset pour les données de capteurs et longueurs d'onde.
    """
    def __init__(self, sensor_data):
        """
        Initialise le dataset avec les données de capteurs.

        :param sensor_data: Dictionnaire contenant les données de capteurs (dict).
        """
        # Liste des noms des capteurs
        self.sensor_names = list(sensor_data.keys())

        # Nombre de données (échantillons)
        self.num_samples = sensor_data[self.sensor_names[0]].shape[0]

        # Nombre de capteurs
        self.num_sensors = len(self.sensor_names)

        # Longueurs d'onde
        self.wavelengths = sensor_data[self.sensor_names[0]][:, 0]

        # Matrice des réponses des capteurs (ligne = un wavelength, colonne = un capteur)
        self.sensor_responses = np.zeros((self.num_samples, self.num_sensors))
        for i, (sensor_name, data) in enumerate(sensor_data.items()):
            self.sensor_responses[:, i] = data[:, 1]

        # Convertir en tenseur PyTorch
        self.wavelengths = torch.FloatTensor(self.wavelengths).view(-1, 1)
        self.sensor_responses = torch.FloatTensor(self.sensor_responses)

        self.wavelengths.to(device)
        self.sensor_responses.to(device)

    def __len__(self):
        """
        Retourne le nombre d'échantillons dans le dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retourne un échantillon du dataset à un indice donné.

        :param idx: Indice de l'échantillon à retourner (int).
        :return: Tuple : (réponses des capteurs, longueur d'onde) pour l'index donné
        """
        return self.sensor_responses[idx], self.wavelengths[idx]

    def get_input_size(self):
        return self.num_sensors

    def get_sensor_names(self):
        return self.sensor_names


# -------- Préparation des DataLoaders pour l'entraînement et le test -------
def prepare_dataloaders(dataset, test_size=0.2, batch_size=32, random_seed=42, shuffle=True):
    """
    Divise un dataset en ensembles d'entraînement et de test.

    :param dataset: Le Dataset complet (CapteursDataset).
    :param test_size: Proportion de données de test (float).
    :param batch_size: Taille des batches pour le DataLoader (int)
    :param random_seed: Graine pour random (int).
    :param shuffle: Mélanger les données (bool).

    :return: Tuple : (train_loader, test_loader)
    """
    # Définir la graine pour la reproductibilité
    torch.manual_seed(random_seed)

    # Taille du dataset de test
    dataset_size = len(dataset)
    test_count = int(dataset_size * test_size)
    train_count = dataset_size - test_count

    # Diviser le dataset en ensembles d'entraînement et de test
    train_dataset, test_dataset = random_split(dataset, [train_count, test_count],
                                               generator=torch.Generator().manual_seed(random_seed))


    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,  # Dataset d'entraînement
        batch_size=batch_size,  # Taille des batches pour calcul des gradients (pas sur tout le dataset), à ajuster au besoin
        shuffle=shuffle,  # Mélanger les données d'entraînement à chaque époque
        num_workers=0,  # Augmenter pour paralléliser le chargement des données
        pin_memory=True  # utile pour un transfert plus rapide vers le GPU
    )

    test_loader = DataLoader(
        test_dataset,  # Dataset de test
        batch_size=batch_size,  # Taille des batches pour calcul des gradients (pas sur tout le dataset)
        shuffle=False,  # Pas besoin de mélanger les données de test
        num_workers=0,  # Augmenter pour paralléliser le chargement des données
        pin_memory=True  # utile pour un transfert plus rapide vers le GPU
    )

    return train_loader, test_loader


# ------------------------- Réseau de neurones -------------------------
class WavelengthPredictor(nn.Module):
    """
    Réseau de neurones pour prédire la longueur d'onde à partir des réponses des capteurs.
    """
    def __init__(self, dropout_rate):
        super(WavelengthPredictor, self).__init__()

        num_sensors = 8 # Nombre de capteurs (input size)

        #hidden_layer_sizes = (512, 512)

        #dropout_rate = 0.2 # Taux de dropout pour régularisation

        # Définition des couches
        self.layers = nn.Sequential(
            # Première hidden layer
            nn.Linear(in_features=8, out_features=512, bias=True),  # 8 in -> 64 out
            nn.ReLU(),                                           # Fonction d'activation ReLU
            #nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
            nn.Dropout(dropout_rate, inplace=False),             # Dropout pour régularisation

            # Deuxième hidden layer
            nn.Linear(in_features=512, out_features=1024, bias=True),
            nn.ReLU(),
            #nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LayerNorm(1024, eps=1e-05, elementwise_affine=True),
            nn.Dropout(dropout_rate, inplace=False),

            # Troisième hidden layer
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(),
            #nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LayerNorm(512, eps=1e-05, elementwise_affine=True),
            nn.Dropout(dropout_rate, inplace=False),

            # Couche de sortie
            nn.Linear(in_features=512, out_features=1, bias=True)
        )

    def forward(self, x):
        """
        Propagation avant du réseau de neurones.

        :param x: Tenseur d'entrée [batch_size, num_sensors]
        """
        return self.layers(x)


# ----------------------- Entraînement du modèle -----------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs = 100, patience=10):
    """
    Entraîne le modèle sur les données d'entraînement.

    :param model: Modèle à entraîner (WavelengthPredictor).
    :param train_loader: DataLoader pour les données d'entraînement.
    :param test_loader: DataLoader pour les données de test.
    :param criterion: Fonction de perte (MSELoss ou autre).
    :param optimizer: Optimiseur (Adam ou autre).
    :param epochs: Nombre d'époques pour l'entraînement.
    :param patience: Nombre d'époques sans amélioration avant d'arrêter l'entraînement.
    """
    #device = "cpu"

    size = len(train_loader.dataset)  # Taille du dataset d'entraînement

    # Scheduler pour ajuster le taux d'apprentissage
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)


    train_losses = []   # Liste pour stocker les pertes d'entraînement
    test_losses = []    # Liste pour stocker les pertes de test
    train_maes = []     # Liste pour stocker les MAEs d'entraînement
    test_maes = []      # Liste pour stocker les MAEs de test

    # Pour early stopping
    best_test_loss = float('inf')    # Meilleure perte de test
    best_model_state = None     # État du modèle avec la meilleure perte
    epochs_no_improve = 0       # Compteur d'époques sans amélioration

    # Calcul du temps d'exécution pour l'entraînement
    start_train_time = perf_counter()

    for epoch in range(epochs):
        # Model en mode entraînement
        model.train()
        running_loss = 0.0  # Initialiser la perte courante
        running_mae = 0.0   # Initialiser l'erreur absolue moyenne courante

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()        # Mettre à jour les poids
            optimizer.zero_grad()   # Réinitialiser les gradients

            # Accumuler la perte
            running_loss += loss.item() * inputs.size(0)

            # Calculer l'erreur absolue moyenne
            # mae = torch.mean(torch.abs(outputs - targets))
            running_mae += torch.sum(torch.abs(outputs - targets)).item()

        # Calculer la perte et MAE moyennes
        epoch_train_loss = running_loss / size
        epoch_train_mae = running_mae / size

        train_losses.append(epoch_train_loss)
        train_maes.append(epoch_train_mae)

        # Évaluation sur le dataset de test
        model.eval()         # Mettre le modèle en mode évaluation
        running_loss = 0.0   # Initialiser la perte courante
        running_mae = 0.0    # Initialiser l'erreur absolue moyenne courante

        with torch.no_grad():  # Pas besoin de calculer les gradients
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Accumuler la perte
                running_loss += loss.item() * inputs.size(0)

                # Calculer l'erreur absolue moyenne
                # mae = torch.mean(torch.abs(outputs - targets))
                running_mae += torch.sum(torch.abs(outputs - targets)).item()

            # Calculer la perte et MAE moyennes
            epoch_test_loss = running_loss / len(test_loader.dataset)
            epoch_test_mae = running_mae / len(test_loader.dataset)

            test_losses.append(epoch_test_loss)
            test_maes.append(epoch_test_mae)

        # Scheduler pour ajuster le taux d'apprentissage
        scheduler.step(epoch_test_loss)


        # Afficher la progression
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Époque {epoch + 1}/{epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.2f} nm - "
                  f"Test Loss: {epoch_test_loss:.4f}, Test MAE: {epoch_test_mae:.2f} nm")

        # Vérifier s'il y a une amélioration
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping si aucune amélioration pendant "patience" époques
        if epochs_no_improve >= patience:
            print(
                f"Arrêt anticipé à l'époque {epoch + 1} car aucune amélioration pendant {patience} époques.")
            break

    # Temps total d'entraînement
    end_train_time = perf_counter()
    print(f"Temps d'entraînement total : {end_train_time - start_train_time:.2f} secondes")

    # Charger le meilleur état du modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_mae': train_maes,
        'test_mae': test_maes
    }

    return model, history


# ------------------- Visualiser l'historique d'entraînement -------------------
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Tracé de la perte (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Entraînement')
    plt.plot(history['test_loss'], label='Validation')
    plt.title('Perte MSE')
    plt.xlabel('Époque')
    plt.ylabel('Erreur')
    plt.legend()
    plt.grid(True)

    # Tracé de l'erreur absolue moyenne
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Entraînement')
    plt.plot(history['test_mae'], label='Validation')
    plt.title('Erreur absolue moyenne')
    plt.xlabel('Époque')
    plt.ylabel('MAE (nm)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------ Visualiser les résultats du test ------------------------
def plot_test_results(model, test_loader):
    """
    Trace un graphique comparant les prédictions du modèle avec les valeurs réelles
    sur l'ensemble de test.

    :param model: Le modèle entraîné
    :param test_loader: DataLoader contenant les données de test
    """
    # Mettre le modèle en mode évaluation
    model.eval()

    # Listes pour stocker les prédictions et les valeurs réelles
    predictions = []
    actual_values = []

    # device = 'cpu'

    # Désactiver le calcul des gradients pour l'inférence
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Ajouter les prédictions et les valeurs réelles aux listes
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())

    # Convertir en arrays numpy
    predictions = np.array(predictions).flatten()
    actual_values = np.array(actual_values).flatten()

    # Calculer la performance globale : un point est bon si l'erreur est inferieur
    # à 10% de la valeur réelle.
    performance = np.abs(predictions - actual_values) / actual_values < 0.1
    performance = np.mean(performance) * 100  # En pourcentage
    print(f"Performance globale : {performance:.2f}% des prédictions sont dans les 10% de l'erreur")

    # Calculer l'erreur absolue pour chaque point
    absolute_errors = np.abs(predictions - actual_values)

    # Calculer les statistiques d'erreur
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean((predictions - actual_values) ** 2))

    # Créer la figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6),
                                   gridspec_kw={'height_ratios': [3, 2]})

    # 1er graph : Prédictions vs Valeurs réelles
    ax1.scatter(actual_values, predictions, alpha=0.5, s=10)

    # Tracer la ligne de référence y=x (prédiction parfaite)
    min_val = min(np.min(predictions), np.min(actual_values))
    max_val = max(np.max(predictions), np.max(actual_values))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', label='Prédiction parfaite')
    ax1.set_xlabel('Longueur d\'onde réelle (nm)')
    ax1.set_ylabel('Longueur d\'onde prédite (nm)')
    ax1.set_xlim(250, 2500)
    ax1.set_title(f'Prédictions vs Valeurs réelles\nMAE: {mae:.2f} nm, RMSE: {rmse:.2f} nm')
    ax1.grid(True)
    ax1.legend()

    # 2ème graph : Erreur absolue selon la longueur d'onde
    # Trier par longueur d'onde
    sort_idx = np.argsort(actual_values)
    sorted_wavelengths = actual_values[sort_idx]
    sorted_errors = absolute_errors[sort_idx]

    # Lissage pour une courbe plus fluide
    window_size = 15
    if len(sorted_wavelengths) > window_size:
        smoothed_errors = np.convolve(sorted_errors, np.ones(window_size) / window_size,
                                      mode='valid')
        smoothed_wavelengths = sorted_wavelengths[window_size - 1:]
    else:
        smoothed_errors = sorted_errors
        smoothed_wavelengths = sorted_wavelengths

    # Tracer la courbe avec aire remplie
    ax2.plot(smoothed_wavelengths, smoothed_errors, linewidth=2)
    ax2.fill_between(smoothed_wavelengths, 0, smoothed_errors, alpha=0.3,
                     color='lightblue', label='Erreur absolue')
    # Courbe d'erreur acceptable
    ax2.plot(smoothed_wavelengths, smoothed_wavelengths * 0.1, 'r-', label='Erreur acceptable (10%)')
    ax2.axhline(y=mae, color='k', linestyle='-', label=f'MAE moyenne: {mae:.2f} nm')
    ax2.set_xlabel('Longueur d\'onde (nm)')
    ax2.set_ylabel('Erreur absolue (nm)')
    ax2.set_title('Erreur absolue en fonction de la longueur d\'onde')
    ax2.set_xlim(250, 2500)
    ax2.set_ylim(0, 300)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()



# ------------------------------ Utilisation ------------------------------
if __name__ == '__main__':
    #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")
    print("\n")

    # Mesurer le temps d'exécution (début)
    start_total_time = perf_counter()

    # Prétraitement des données
    reponses_capteurs = DataPreProcess(plastic_name='Petri')
    reponses_capteurs = reponses_capteurs.all_sensors

    # Créer le dataset avec les données des capteurs
    dataset = CapteursDataset(reponses_capteurs)

    # Paramètres
    batch_size = 128                         # Taille des batches pour le DataLoader
    test_size = 0.2                         # Proportion de données de test
    random_seed = 42                        # Graine pour la reproductibilité
    learning_rate = 0.0031701443704886616   # Taux d'apprentissage
    dropout_rate = 0.11276492753388213      # Taux de dropout pour régularisation
    weight_decay = 2.3936000236956148e-05   # Régularisation L2
    num_epochs = 500                        # Nombre d'époques pour l'entraînement
    patience = 30                           # Nombre d'époques sans amélioration avant d'arrêter l'entraînement

    # Diviser les données et créer les DataLoaders
    train_loader, test_loader = prepare_dataloaders(dataset,
                                                    test_size=test_size,
                                                    batch_size=batch_size,
                                                    random_seed=random_seed,
                                                    shuffle=True
                                                    )

    # Initialiser le modèle
    model = WavelengthPredictor(dropout_rate=dropout_rate).to(device)
    print("----------------------------------")
    print(f"Modèle : {model}")
    print("----------------------------------")

    # Définir la fonction de perte et l'optimiseur
    #criterion = nn.MSELoss()  # Fonction de perte MSE
    criterion = nn. HuberLoss(delta=10)  # Fonction de perte Huber
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Optimiseur Adam

    # Entraîner le modèle
    trained_model, history = train_model(model,
                                 train_loader,
                                 test_loader,
                                 criterion,
                                 optimizer,
                                 epochs=num_epochs,
                                 patience=patience
                                 )

    #torch.save(trained_model.state_dict(), "modele_nn_pytroch.pt")
    # torch.save(model.state_dict(), 'model_nn_pytorch_weights.pth')

    # Temps total d'exécution
    print(f"Temps d'exécution total : {perf_counter() - start_total_time}")

    # Visualiser l'historique d'entraînement
    plot_history(history)

    # Visualiser les résultats du test
    plot_test_results(trained_model, test_loader)