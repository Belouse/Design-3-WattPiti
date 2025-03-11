import tkinter as tk
from tkinter import ttk
import os
from PIL import ImageTk, Image

class InterfaceWattpiti(tk.Tk):
    def __init__(self):
        super().__init__()

        #création de l'interface
        self.title("Puissance-mètre Wattpiti")
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
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
        self.configFrame = ttk.Frame(self, width=400, height=200, borderwidth=3, style='frameLabelStyle.TLabelframe')
        self.configFrame.grid(row=0, column=1, pady=5, padx=5, sticky="nsew")
        self.configFrame.grid_propagate(False)

        # Création d'un titre pour les paramètres de configuration
        self.labelConfig = ttk.Label(self.configFrame, text="Configuration de la simulation", style = 'frameLabelStyle.TLabelframe.Label')
        self.labelConfig.grid(row=0, column=1, padx=5, pady=5, sticky = "nsew")

        #Création d'un bouton pour démarrer la simulation
        startButtonStyle = ttk.Style()
        startButtonStyle.configure("startButtonStyle.TButton", background = "#00FF00", relief = "raised", font = ("Inter", 10, "bold"))
        self.startButton = ttk.Button(self, text="Commencer",style="startButtonStyle.TButton", command=self.click_start)
        self.startButton.place(x = 470, y= 150)

        #Création d'un bouton pour arrêter la simulation
        stopButtonStyle = ttk.Style()
        stopButtonStyle.configure("stopButtonStyle.TButton", background = "red", relief = "raised",  font = ("Inter", 10, "bold"))
        self.stopButton = ttk.Button(self, text="Arrêt", style="stopButtonStyle.TButton", command=self.click_stop)
        self.stopButton.place(x=270, y=150)

        #Création d'un bouton pour réinitialiser la simulation
        resetButtonStyle = ttk.Style()
        resetButtonStyle.configure("resetButtonStyle.TButton", background = "yellow", relief = "raised",  font = ("Inter", 10, "bold"))
        self.resetButton = ttk.Button(self, text='Reset', style = "resetButtonStyle.TButton", command=self.click_reset)
        self.resetButton.place(x=370, y=150)

        #Création d'une entrée pour la fréquence d'échantillonnage
        freqVar = tk.StringVar()
        freqVar.trace_add("write", self.freq_value)
        self.freqEntry = ttk.Entry(self, textvariable = freqVar)
        self.freqEntry.place(x =445, y = 50)
        
        #Création d'un label pour la fréquence d'échantillonnage
        self.labelFreq = ttk.Label(self, text="Fréquence d'échantillonnage (Hz):", style = "labelStyle.TLabel")
        self.labelFreq.place(x=250, y = 50)

        #Création d'une entrée pour le temps d'acquisition
        timeVar = tk.StringVar()
        timeVar.trace_add("write", self.time_value)
        self.timeEntry = ttk.Entry(self, textvariable = timeVar)
        self.timeEntry.place(x= 445, y=100)

        #Création d'un label pour le temps d'acquisition
        self.labelTime = ttk.Label(self, text="Temps d'acquisition (s):", style= "labelStyle.TLabel")
        self.labelTime.place(x = 250, y = 100)



        # Création d'un cadre pour les options d'enregistrement
        self.optionsFrame = ttk.Frame(self, width=460, height=200, borderwidth=3, style='frameLabelStyle.TLabelframe')
        self.optionsFrame.grid(row=0, column=2, pady=5, padx=5, sticky="nsew", columnspan = 2)
        self.optionsFrame.grid_propagate(False)

        # Création d'un titre pour les options d'enregistrement
        self.labelOptions = ttk.Label(self.optionsFrame, text="Options d'enregistrement", style='frameLabelStyle.TLabelframe.Label')
        self.labelOptions.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Création d'une entrée pour le nom du fichier
        self.fileNameVar = tk.StringVar()
        self.fileNameVar.trace_add("write", self.save_data)
        self.fileName = ttk.Entry(self, textvariable = self.fileNameVar)
        self.fileName.place(x = 800, y =50)

        #Création d'un label pour le nom du fichier
        self.labelFileName = ttk.Label(self, text = "Nom du fichier:", style = "labelStyle.TLabel")
        self.labelFileName.place(x=670, y=50)

        #Création d'un radio bouton pour le format de fichier
        formatVar = tk.StringVar()
        self.fileFormatCsv = tk.Radiobutton(self, text = ".CSV", variable = formatVar, value = 1, command = self.file_format, background = "#DCDCDC")
        self.fileFormatXlsx = tk.Radiobutton(self, text=".XLSX", variable = formatVar, value = 2, command = self.file_format, background = "#DCDCDC")
        self.fileFormatCsv.place(x=800, y=100)
        self.fileFormatXlsx.place(x=900, y=100)

        #Création d'un label pour les radio buttons du format du fichier
        self.labelFormatChoice = ttk.Label(self, text = "Format du fichier:", style = "labelStyle.TLabel")
        self.labelFormatChoice.place(x=670, y=103)
        
        #Création d'un label pour le choix des données à enregistrer
        self.LabelDataChoice = ttk.Label(self, text = "Données à enregistrer:", style = "labelStyle.TLabel")
        self.LabelDataChoice.place(x = 670, y= 130)


        #Création d'un checkbox pour le choix des données à enregistrer
        #Longueur d'onde
        wavelengthCheckButtonBool = tk.BooleanVar()
        self.wavelengthCheckButton = ttk.Checkbutton(self, text= "Longueur d'onde", variable = wavelengthCheckButtonBool)
        self.wavelengthCheckButton.place(x=810, y=150)
        #Puissance
        powerCheckButtonBool = tk.BooleanVar()
        self.powerCheckButton = ttk.Checkbutton(self, text="Puissance", variable = powerCheckButtonBool)
        self.powerCheckButton.place(x=810, y = 130)
        #Position
        positionCheckButtonBool = tk.BooleanVar()
        self.positionCheckButton = ttk.Checkbutton(self, text="Position", variable = positionCheckButtonBool)
        self.positionCheckButton.place(x = 810, y = 170)

        #Création d'un bouton pour enregister les données
        saveButtonStyle = ttk.Style()
        saveButtonStyle.configure("saveButtonStyle.TButton", background = "#B2DEF7", relief = "raised", font = ("Inter", 10, "bold"))
        self.saveButton = ttk.Button(self, text="Enregistrer", style="saveButtonStyle.TButton")
        self.saveButton.place(x= 990, y = 150)




        #Création d'un frame pour les résultats de la simulation
        self.resultsFrame = ttk.Frame(self, width=250, height=200, borderwidth=3, style = "frameLabelStyle.TLabelframe")
        self.resultsFrame.grid(row = 0, column = 4, padx=5, pady=5, sticky = "nsew")
        self.resultsFrame.grid_propagate(False)

        #Création d'un label pour les résultats de la simulation
        self.resultsLabel = ttk.Label(self.resultsFrame, text="Résultats de la simulation",style='frameLabelStyle.TLabelframe.Label')
        self.resultsLabel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")


        #Création d'un display pour la puissance instantanée
        self.powerVar = tk.StringVar()
        self.powerVar.set("00.00")

        self.powerDisplayLabel = ttk.Label(self, text= "Puissance instantanée (W):", font = ("Inter", 14, "bold"))
        self.powerDisplayLabel.place(x=1150, y=58)                                                           
        self.powerDisplay = ttk.Label(self, textvariable = self.powerVar , font = ("Inter", 24, "bold"))
        self.powerDisplay.place(x=1430, y =50)

        #Création d'un display pour la longueur d'onde
        self.wavelenghtVar = tk.StringVar()
        self.wavelenghtVar.set("000.0")

        self.wavelenghtDisplayLabel = ttk.Label(self, text= "Longueur d'onde (nm):", font = ("Inter", 14, "bold"))
        self.wavelenghtDisplayLabel.place(x=1150, y=108)                                                           
        self.wavelenghtDisplay = ttk.Label(self, textvariable = self.wavelenghtVar , font = ("Inter", 24, "bold"))
        self.wavelenghtDisplay.place(x= 1430, y = 100)

        #Création d'un display pour la position centrale du faisceau
        self.positionXVar = tk.StringVar()
        self.positionXVar.set("0")
        self.positionYVar = tk.StringVar()
        self.positionYVar.set("0")

        self.positionDisplayLabel = ttk.Label(self, text="Position (mm):", font = ("Inter", 14, "bold"))
        self.positionDisplayLabel.place(x = 1150, y = 158)
        
        self.positionXDisplay = ttk.Label(self, textvariable= self.positionXVar, font = ("Inter", 24, "bold"))
        self.positionXDisplay.place(x = 1400, y= 150)
        self.positionYDisplay = ttk.Label(self, textvariable=self.positionYVar, font = ("Inter", 24, "bold"))
        self.positionYDisplay.place(x = 1500, y= 150)

        self.positionXLabel = ttk.Label(self, text = "x :", font= ("Inter", 24, "bold"))
        self.positionXLabel.place(x = 1350, y = 149)

        self.positionYLabel = ttk.Label(self, text = "y :", font= ("Inter", 24, "bold"))
        self.positionYLabel.place(x = 1450, y = 149)


        ###Création d'un graphique de la puissance en fonction du temps
        self.powerPlotFrame = ttk.Frame(self, width=1100, height=600, borderwidth=3, style = "frameLabelStyle.TLabelframe")
        self.powerPlotFrame.grid(row = 2, column= 0 , columnspan = 3, rowspan = 1, padx = 5, pady = 5, sticky = "nsew")

        self.powerPlotLabel = ttk.Label(self, text= "Graphique de la puissance en fonction du temps", style = "frameLabelStyle.TLabelframe.Label")
        self.powerPlotLabel.place(x = 10, y = 230)


        ###Création d'un graphique de la position centrale du faisceau
        self.posPlotFrame = ttk.Frame(self, width=793, height=600, borderwidth=3, style = "frameLabelStyle.TLabelframe")
        self.posPlotFrame.grid(row = 2, column= 4 , columnspan = 3, rowspan = 1, padx = 5, pady = 5, sticky = "nsew")

        self.posPlotLabel = ttk.Label(self, text= "Position centrale du faisceau", style = "frameLabelStyle.TLabelframe.Label")
        self.posPlotLabel.place(x = 1130, y = 230)






    #Fonction du bouton pour démarrer la simulation
    def click_start(self):
        pass

    #Fonction du bouton pour arrêter la simulation
    def click_stop(self):
        pass

    #Fontion du bouton pour réinitialiser la simulation
    def click_reset(self):
        pass

    #Fonction pour la fréquence d'échantillonnage
    def freq_value(self):
        pass

    #Fonction pour le temps d'acquisition
    def time_value(self):
        pass

    #Fonction pour le nom du fichier
    def save_data(self):
        pass
    
    #Fonction pour le format du fichier enregistré
    def file_format(self):
        pass
    
    #Fonction pour la valeur de puissance lue
    def power_value(self):
        pass

    #Fonction pour la valeur de la longueur d'onde lue
    def wavelenght_value(self):
        pass

    #Fonction pour la valeur de la position lue
    def position_value(self):
        pass

if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()