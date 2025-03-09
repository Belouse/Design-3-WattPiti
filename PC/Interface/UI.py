import tkinter as tk
from tkinter import ttk
import os
from PIL import ImageTk, Image

class InterfaceWattpiti(tk.Tk):
    def __init__(self):
        super().__init__()

        #création de l'interface
        self.title("Puissance-mètre Wattpiti")
        self.geometry("1800x1800")
        self.configure(background="white")


        mystyle = ttk.Style()
        mystyle.theme_use('alt')   # choose other theme
        mystyle.configure('MyStyle.TLabelframe', background = "white", borderwidth=10, relief='solid', labelmargins=20)
        mystyle.configure('MyStyle.TLabelframe.Label', font=('Inter', 11, 'bold'))


        #Création de la grille pour le logo
        self.viewLogo = ttk.Frame(self, width=200, height=200)
        self.viewLogo.grid(row=0, column=0, pady=5, padx=5, sticky="nsew")
        self.viewLogo.grid_propagate(True)

        #Importer le logo de la compagnie et l'insérer dans la grille
        dir = os.path.dirname(__file__)
        image_path = os.path.join(dir, "logoWattpiti.jpg")
        self.logo = ImageTk.PhotoImage(Image.open(image_path))
        self.labelLogo = ttk.Label(self, image = self.logo)
        self.labelLogo.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        #Changer la taille du logo de la compagnie
        self.logo = Image.open(image_path)
        self.logo = self.logo.resize((200, 200))
        self.logo = ImageTk.PhotoImage(self.logo)
        self.labelLogo.configure(image=self.logo)

        #Création d'un cadre pour les paramètres de configuration
        self.configFrame = ttk.Frame(self, width=200, height=350, borderwidth=3, style='MyStyle.TLabelframe')
        self.configFrame.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        self.configFrame.grid_propagate(False)

        # Création d'un titre pour les paramètres de configuration
        self.labelConfig = ttk.Label(self.configFrame, text="Configuration", style = 'MyStyle.TLabelframe.Label')
        self.labelConfig.grid(row=0, column=1, padx=5, pady=5, sticky = "nsew")

        #Création d'un bouton pour démarrer la simulation
        self.startButton = ttk.Button(self, text="Commencer", command=self.click_start)
        self.startButton.place(x = 65, y= 280)

        #Création d'un bouton pour arrêter la simulation
        self.stopButton = ttk.Button(self, text="Arrêt", command=self.click_stop)
        self.stopButton.place(x=65, y=320)

        #Création d'un bouton pour réinitialiser la simulation
        self.resetButton = ttk.Button(self, text='Reset', command=self.click_reset)
        self.resetButton.place(x=65, y=360)

        #Création d'un label pour la fréquence d'échantillonnage
        self.labelFreq = ttk.Label(self, text="Fréquence d'échantillonnage (Hz)")
        self.labelFreq.place(x=64, y = 400)

        # Création d'un cadre pour les options d'enregistrement
        self.optionsFrame = ttk.Frame(self, width=200, height=350, borderwidth=3, style='MyStyle.TLabelframe')
        self.optionsFrame.grid(row=2, column=0, pady=5, padx=5, sticky="nsew")
        self.optionsFrame.grid_propagate(False)

        # Création d'un titre pour les options d'enregistrement
        self.labelOptions = ttk.Label(self.optionsFrame, text="Options d'enregistrement", style='MyStyle.TLabelframe.Label')
        self.labelOptions.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")


    #Fonction du bouton pour démarrer la simulation
    def click_start(self):
        pass

    #Fonction du bouton pour arrêter la simulation
    def click_stop(self):
        pass

    #Fontion du bouton pour réinitialiser la simulation
    def click_reset(self):
        pass

if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()