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

        #Style des FrameLabels pour l'interface
        frameLabelStyle = ttk.Style()
        frameLabelStyle.theme_use('default')   # choose other theme
        frameLabelStyle.configure('frameLabelStyle.TLabelframe', borderwidth=10)
        frameLabelStyle.configure('frameLabelStyle.TLabelframe.Label', font=('Inter', 11, 'bold'))

        #Style des labels pour l'interface
        labelStyle = ttk.Style()
        labelStyle.configure("labelStyle.TLabel", font=("Inter", 9, "bold"))


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
        self.configFrame = ttk.Frame(self, width=800, height=200, borderwidth=3, style='frameLabelStyle.TLabelframe')
        self.configFrame.grid(row=0, column=1, pady=5, padx=5, sticky="nsew")
        self.configFrame.grid_propagate(False)

        # Création d'un titre pour les paramètres de configuration
        self.labelConfig = ttk.Label(self.configFrame, text="Configuration", style = 'frameLabelStyle.TLabelframe.Label')
        self.labelConfig.grid(row=0, column=1, padx=5, pady=5, sticky = "nsew")

        #Création d'un bouton pour démarrer la simulation
        startButtonStyle = ttk.Style()
        startButtonStyle.configure("startButtonStyle.TButton", background = "#00FF00", relief = "raised")
        self.startButton = ttk.Button(self, text="Commencer",style="startButtonStyle.TButton", command=self.click_start)
        self.startButton.place(x = 450, y= 100)

        #Création d'un bouton pour arrêter la simulation
        stopButtonStyle = ttk.Style()
        stopButtonStyle.configure("stopButtonStyle.TButton", background = "red", relief = "raised")
        self.stopButton = ttk.Button(self, text="Arrêt", style="stopButtonStyle.TButton", command=self.click_stop)
        self.stopButton.place(x=350, y=100)

        #Création d'un bouton pour réinitialiser la simulation
        resetButtonStyle = ttk.Style()
        resetButtonStyle.configure("resetButtonStyle.TButton", background = "yellow", relief = "raised")
        self.resetButton = ttk.Button(self, text='Reset', style = "resetButtonStyle.TButton", command=self.click_reset)
        self.resetButton.place(x=250, y=100)

        #Création d'une entrée pour la fréquence d'échantillonnage
        freqVar = tk.StringVar()
        freqVar.trace_add("write", self.freq)
        self.freqEntry = ttk.Entry(self, textvariable = freqVar)
        self.freqEntry.place(x =760, y = 50)
        
        #Création d'un label pour la fréquence d'échantillonnage
        self.labelFreq = ttk.Label(self, text="Fréquence d'échantillonnage (Hz):", style = "labelStyle.TLabel")
        self.labelFreq.place(x=565, y = 50)

        #Création d'une entrée pour le temps d'acquisition
        timeVar = tk.StringVar()
        timeVar.trace_add("write", self.time)
        self.timeEntry = ttk.Entry(self, textvariable = timeVar)
        self.timeEntry.place(x= 760, y=100)

        #Création d'un label pour le temps d'acquisition
        self.labelTime = ttk.Label(self, text="Temps d'acquisition (s):", style= "labelStyle.TLabel")
        self.labelTime.place(x = 565, y = 100)


        # Création d'un cadre pour les options d'enregistrement
        self.optionsFrame = ttk.Frame(self, width=200, height=350, borderwidth=3, style='frameLabelStyle.TLabelframe')
        self.optionsFrame.grid(row=2, column=0, pady=5, padx=5, sticky="nsew")
        self.optionsFrame.grid_propagate(False)

        # Création d'un titre pour les options d'enregistrement
        self.labelOptions = ttk.Label(self.optionsFrame, text="Options d'enregistrement", style='frameLabelStyle.TLabelframe.Label')
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

    #Fontion pour la fréquence d'échantillonnage
    def freq(self):
        pass

    #Fontion pour le temps d'acquisition
    def time(self):
        pass

if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()