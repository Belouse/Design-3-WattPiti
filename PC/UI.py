import tkinter as tk
from tkinter import ttk
import os
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import numpy as np
import time
from DataContainerClass import DataContainer
from AlgoPosition import AlgoPosition
from AlgorithmManagerClass import AlgorithmManager
from SerialManagerClass import SerialManager
import serial.tools.list_ports
from datetime import datetime
from SerialListenerClass import SerialListener


class InterfaceWattpiti(tk.Tk):
    def __init__(self):
        super().__init__()

        #Donnée du temps
        #self.startTime = time.time()

        #Création d'une instance de la classe DataContainer pour stocker les données
        self.dataContainer = DataContainer() #Instance de la classe DataContainer
        self.algorithmManager = AlgorithmManager(self.dataContainer) #Instance de la classe AlgorithmManager
        self.algoPosition = AlgoPosition() #Instance de la classe Algoposition (pour le graphique de la position)
        self.serialManager = SerialManager(self.dataContainer, maxData=1) #Instance de la classe SerialManager
        

        #création de l'interface
        self.title("Puissance-mètre Wattpiti")
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        #self.geometry("1000x1000")
        self.configure(background="white")

        #Style des FrameLabels pour l'interface
        frameLabelStyle = ttk.Style()
        frameLabelStyle.theme_use('default')
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
        self.labelConfig = ttk.Label(self.configFrame, text="Configuration de la prise de données", style = 'frameLabelStyle.TLabelframe.Label')
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


        #Création d'une liste déroulante pour choisir le port
        self.ports = serial.tools.list_ports.comports() #Lister les ports disponibles dans l'ordinateur
        self.portList = []
        for port in self.ports: #Affiche les ports et leur description dans l'interface
            if "Bluetooth" not in port.description: #Enlever les ports Bluetooth de la liste
                self.portList.append(f"{port.device}, {port.description}")

        self.selected_port = tk.StringVar()
        if len(self.portList) == 1: #Si un seul port est disponible, le sélectionner par défaut
            self.selected_port.set(self.portList[0])
        self.portComboBox = ttk.Combobox(self, values=self.portList, width=35, textvariable= self.selected_port)
        self.portComboBox.place(x=350, y=50)

        #Création d'un label pour le choix du port
        self.labelFreq = ttk.Label(self, text="Choix du port:", style = "labelStyle.TLabel")
        self.labelFreq.place(x=250, y = 50)


        #Choix de la longueur d'onde
        self.waveLenghtList = ["À déterminer", "450", "976", "1976"]
        self.waveLenghtComboBox = ttk.Combobox(self, values=self.waveLenghtList, width=20)
        self.waveLenghtComboBox.place(x=440, y=80)
        self.waveLenghtComboBox.current(0) 

        #Label pour le choix de la longueur d'onde
        self.labelWaveLength = ttk.Label(self, text="Longueur d'onde (nm):", style = "labelStyle.TLabel")
        self.labelWaveLength.place(x=250, y=80)



        # Création d'un cadre pour les options d'enregistrement
        self.optionsFrame = ttk.Frame(self, width=400, height=200, borderwidth=3, style='frameLabelStyle.TLabelframe')
        self.optionsFrame.grid(row=0, column=2, pady=5, padx=5, sticky="nsew", columnspan = 2)
        self.optionsFrame.grid_propagate(False)

        # Création d'un titre pour les options d'enregistrement
        self.labelOptions = ttk.Label(self.optionsFrame, text="Options d'enregistrement", style='frameLabelStyle.TLabelframe.Label')
        self.labelOptions.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Création d'une entrée pour le nom du fichier
        self.fileNameVar = tk.StringVar()
        self.fileName = ttk.Entry(self, textvariable = self.fileNameVar)
        self.fileName.place(x = 800, y =50)

        #Création d'un label pour le nom du fichier
        self.labelFileName = ttk.Label(self, text = "Nom du fichier:", style = "labelStyle.TLabel")
        self.labelFileName.place(x=670, y=50)

        #Création d'un label pour le choix des données à enregistrer
        self.LabelDataChoice = ttk.Label(self, text = "Données à enregistrer:", style = "labelStyle.TLabel")
        self.LabelDataChoice.place(x = 670, y= 80)
                
        


        #Création d'un checkbox pour le choix des données à enregistrer
        #Longueur d'onde
        self.wavelengthCheckButtonBool = tk.BooleanVar()
        self.wavelengthCheckButton = ttk.Checkbutton(self, text= "Longueur d'onde", variable = self.wavelengthCheckButtonBool)
        self.wavelengthCheckButton.place(x=810, y=110)
        self.wavelengthCheckButtonBool.set(True)
        #Puissance
        self.powerCheckButtonBool = tk.BooleanVar()
        self.powerCheckButton = ttk.Checkbutton(self, text="Puissance", variable = self.powerCheckButtonBool)
        self.powerCheckButton.place(x=810, y = 90)
        self.powerCheckButtonBool.set(True)
        #Position
        self.positionCheckButtonBool = tk.BooleanVar()
        self.positionCheckButton = ttk.Checkbutton(self, text="Position", variable = self.positionCheckButtonBool)
        self.positionCheckButton.place(x = 810, y = 130)
        self.positionCheckButtonBool.set(True)

        #Création d'un bouton pour enregister les données
        self.saveButtonStyle = ttk.Style()
        self.saveButtonStyle.configure("saveButtonStyle.TButton", background = "#B2DEF7", relief = "raised", font = ("Inter", 10, "bold"))
        self.saveButton = ttk.Button(self, text="Enregistrer", style="saveButtonStyle.TButton", command = self.save_data)
        self.saveButton.place(x= 930, y = 150)




        #Création d'un frame pour les résultats de la simulation
        self.resultsFrame = ttk.Frame(self, width=480, height=200, borderwidth=3, style = "frameLabelStyle.TLabelframe")
        self.resultsFrame.grid(row = 0, column = 4, padx=5, pady=5, sticky = "nsew")
        self.resultsFrame.grid_propagate(False)

        #Création d'un label pour les résultats de la simulation
        self.resultsLabel = ttk.Label(self.resultsFrame, text="Résultats de la prise de données",style='frameLabelStyle.TLabelframe.Label')
        self.resultsLabel.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")


        #Création d'un display pour la puissance instantanée
        self.powerVar = tk.StringVar() #Variable de la puissance
        self.powerVar.set("00.00")

        self.powerDisplayLabel = ttk.Label(self, text= "Puissance instantanée (W):", font = ("Inter", 14, "bold"))
        self.powerDisplayLabel.place(x=1050, y=58)                                                           
        self.powerDisplay = ttk.Label(self, textvariable = self.powerVar , font = ("Inter", 24, "bold"))
        self.powerDisplay.place(x=1380, y =50)

        #Création d'un display pour la longueur d'onde
        self.wavelenghtVar = tk.StringVar()
        self.wavelenghtVar.set("000.0")

        self.wavelenghtDisplayLabel = ttk.Label(self, text= "Longueur d'onde (nm):", font = ("Inter", 14, "bold"))
        self.wavelenghtDisplayLabel.place(x=1050, y=108)                                                           
        self.wavelenghtDisplay = ttk.Label(self, textvariable = self.wavelenghtVar , font = ("Inter", 24, "bold"))
        self.wavelenghtDisplay.place(x= 1380, y = 100)

        #Création d'un display pour la position centrale du faisceau
        self.positionXVar = tk.StringVar() #Variable de la position sur l'axe x
        self.positionXVar.set("0")
        self.positionYVar = tk.StringVar() #Variable de la position sur l'axe y
        self.positionYVar.set("0")

        self.positionDisplayLabel = ttk.Label(self, text="Position (mm):", font = ("Inter", 14, "bold"))
        self.positionDisplayLabel.place(x = 1050, y = 158)
        
        self.positionXDisplay = ttk.Label(self, textvariable= self.positionXVar, font = ("Inter", 24, "bold"))
        self.positionXDisplay.place(x = 1280, y= 150)
        self.positionYDisplay = ttk.Label(self, textvariable=self.positionYVar, font = ("Inter", 24, "bold"))
        self.positionYDisplay.place(x = 1420, y= 150)

        self.positionXLabel = ttk.Label(self, text = "x :", font= ("Inter", 24, "bold"))
        self.positionXLabel.place(x = 1230, y = 149)

        self.positionYLabel = ttk.Label(self, text = "y :", font= ("Inter", 24, "bold"))
        self.positionYLabel.place(x = 1370, y = 149)


        ###Création du frame pour le graphique de la puissance en fonction du temps
        self.powerPlotFrame = ttk.Frame(self, width=925, height=700, borderwidth=3, style = "frameLabelStyle.TLabelframe")
        self.powerPlotFrame.grid(row = 2, column= 0 , columnspan = 3, rowspan = 1, padx = 5, pady = 5, sticky = "nsew")

        self.powerPlotLabel = ttk.Label(self, text= "Graphique de la puissance en fonction du temps", style = "frameLabelStyle.TLabelframe.Label")
        self.powerPlotLabel.place(x = 10, y = 230)

        #Création du graphique pour la puissance en fonction du temps
        self.powerFig = Figure(figsize = (8.9, 5.8), dpi = 100)
        self.axPow = self.powerFig.add_subplot(111)
        self.axPow.set_xlabel("Temps (s)")
        self.axPow.set_ylabel("Puissance (W)")

        self.powerCanvas = FigureCanvasTkAgg(self.powerFig, master = self)
        self.powerCanvas.draw()
        self.powerPlot = self.powerCanvas.get_tk_widget()
        self.powerPlot.place(x=20, y=280)




        ###Création du frame du graphique de la position centrale du faisceau
        self.posPlotFrame = ttk.Frame(self, width=575, height=700, borderwidth=3, style = "frameLabelStyle.TLabelframe")
        self.posPlotFrame.place(x = 940, y = 217)

        self.posPlotLabel = ttk.Label(self, text= "Position centrale du faisceau", style = "frameLabelStyle.TLabelframe.Label")
        self.posPlotLabel.place(x = 950, y = 230)

        #Création du graphique pour la position centrale du faisceau
        self.posFig = Figure(figsize = (5, 5.8), dpi = 100)
        self.axPos = self.posFig.add_subplot(111)
        self.axPos.set_xlabel("Position x (mm)")
        self.axPos.set_ylabel("Position y (mm)")


        self.posCanvas = FigureCanvasTkAgg(self.posFig, master = self)
        self.posPlot = self.posCanvas.get_tk_widget()
        self.posPlot.place(x = 970, y = 280)

        #Création d'un bouton pour calibrer les capteurs
        self.tareButton = ttk.Button(self, text= "Tare", style="saveButtonStyle.TButton", command = self.click_tare)
        self.tareButton.place(x = 420, y = 110)
        self.tareLabel = ttk.Label(self, text = "Calibration des capteurs:", style = "labelStyle.TLabel")
        self.tareLabel.place(x=250, y = 110)


        #Gestion de la loop
        self.running = False
        self.dataArray = []
        self.powArray = []

        #Si les ports sont nuls
        if len(self.portList) == 0:
            self.check_ports()



        self.startTime = time.time()




    #Fonction du bouton pour démarrer la simulation
    def click_start(self):
        try:
        #Importation des classes externes pour stocker les données
            self.serialManager.setPortName(self.selected_port.get().split(",")[0])
            if not self.running:
                self.running = True
                self.loop() 
        except serial.SerialException as e:
            self.error_handling("Erreur de connexion au port série")
            self.running = False


    def loop(self):
        if self.running: #Vérifier si une simulation est en cours
            self.currentTime = time.time() - self.startTime #Calculer le temps écoulé depuis l'ouverture de l'interface

            #Importation des données de dataContainer
            self.serialManager.updateDataFromMCU(1) #Mettre à jour les données de la carte mère
            self.algorithmManager.calculatePosition()
            self.algorithmManager.calculateWavelength()
            self.algorithmManager.calculatePower()
            self.newposition = self.dataContainer.position
            self.newWaveLenght = self.dataContainer.wavelength
            self.newpower = self.dataContainer.power  
            self.rawTemperatureMatrix = self.dataContainer.rawTemperatureMatrix
            self.rawData = self.serialManager.serialListener.readData(1, printExecutionTime=False) #Lire les données du port série
            print(self.rawData)

            #Importer les données dans une liste
            self.dataArray.append((self.currentTime, self.newpower, self.newWaveLenght, self.newposition))

            


            #Formater les données pour les afficher dans l'interface graphique
            self.newpower = "{:.2f}".format(self.newpower) #Formater la puissance
            self.newWaveLenght = "{:.1f}".format(self.newWaveLenght) #Formater la longueur d'onde
            self.newposition = [round(x, 2) for x in self.newposition] #Formater la position centrale du faisceau
            
            #Mettre à jour les labels dans l'interface graphique
            self.powerVar.set(str(self.newpower)) #Puissance
            self.wavelenghtVar.set(str(self.newWaveLenght)) #Longueur d'onde
            self.positionXVar.set(str(self.newposition[0])) #Positon x
            self.positionYVar.set(str(self.newposition[1])) #Position y
            

            #Graphique de la position centrale du faisceau
            self.axPos.clear()
            self.axPos.set_xlabel("Position x (mm)")
            self.axPos.set_ylabel("Position y (mm)")
            AlgoPosition.calculatePosition(self.algoPosition, self.dataContainer)
            contour = self.axPos.contourf(self.dataContainer.interpolatedTemperatureGrid[0], 
                            self.dataContainer.interpolatedTemperatureGrid[1], 
                            self.dataContainer.interpolatedTemperatureGrid[2], 
                            levels=150,
                            cmap='turbo')
            
            for row in self.dataContainer.thermalCaptorPosition: #Affichage de la grille de capteurs sur le graphique
                for (x, y) in row:
                    rect = patches.Rectangle((x - 1.5, y - 1.5), 0.75, 0.75, linewidth=1.5, edgecolor='white', facecolor='none', alpha=0.5)
                    self.axPos.add_patch(rect)
            self.posCanvas.draw()
            self.posCanvas.get_tk_widget().update()

            #Graphique de la puissance en fonction du temps
            self.axPow.clear()
            self.axPow.set_xlabel("Temps (s)")
            self.axPow.set_ylabel("Puissance (W)")
            if len(self.dataArray) > 20: #limiter le nombre de points sur le graphique
                self.axPow.set_xlim(self.dataArray[-20][0], self.dataArray[-1][0])

            self.timeArray = list(zip(* self.dataArray))[0]
            self.powValues = list(zip(* self.dataArray))[1]
            self.axPow.plot(list(self.timeArray), list(self.powValues), color = "blue")

            self.powerCanvas.draw()
            self.powerCanvas.get_tk_widget().update()
            self.check_ports() #Vérifier si le port est toujours connecté
        
            print(self.dataContainer.max_temperature)
            self.loop()


    
    
    
    def click_stop(self): #Permet de mettre sur pause la simulation
        self.running = False
        self.axPos.clear()
        self.axPow.clear()

    #Fontion du bouton pour réinitialiser la simulation
    def click_reset(self):
        if self.running == True:
            self.click_stop()

        #Réinitialiser les variables
        self.dataArray = []
        self.powArray = []
        self.timeArray = []
        self.powerVar.set("00.00")
        self.wavelenghtVar.set("000.0")
        self.positionXVar.set("0")
        self.positionYVar.set("0") 
        self.axPos.clear()
        self.axPow.clear()


        self.click_start() 
        


    def click_tare(self):
        #Fonction pour calibrer les capteurs
        pass

    #Fonction pour enregistrer les données
    def save_data(self):
        self.savingTimeRef = time.time()
        self.date = datetime.now()
        self.dtstring = self.date.strftime("%d-%m-%Y %H_%M_%S")

        if self.running == True:
            self.click_stop()
        
        #Désactiver les widgets de l'interface
        self.disable_widgets()

       
        self.choiceArray = [self.powerCheckButtonBool.get(), self.wavelengthCheckButtonBool.get(), self.positionCheckButtonBool.get()]

        #Enregistrer les données dans un fichier txt
        
        self.file_name = self.fileNameVar.get()
        if self.fileNameVar.get() == "":
            self.file_name = f"{self.dtstring}.csv"

        if not self.file_name.endswith('.csv'):
            self.file_name += '.csv'
        # Créer un dossier "savedData" s'il n'existe pas
        save_dir = os.path.join(os.path.dirname(__file__), "savedData")
        os.makedirs(save_dir, exist_ok=True)
        
        # Chemin complet du fichier
        self.file_name = os.path.join(save_dir, self.file_name)
            

        if self.choiceArray == [True, True, True]:
            with open(self.file_name, 'w') as file:
                file.write("Temps (s),Puissance (W),Longueur d'onde (nm),Position x (mm),Position y (mm)\n")
                for i in self.dataArray:
                    file.write(f"{i[0]},{i[1]},{i[2]},{i[3][0]},{i[3][1]}\n")
        
        elif self.choiceArray == [True, True, False]:
            with open(self.file_name, 'w') as file:
                file.write("Temps (s),Puissance (W),Longueur d'onde (nm)\n")
                for i in self.dataArray:
                    file.write(f"{i[0]},{i[1]},{i[2]}\n")
        
        elif self.choiceArray == [True, False, True]:
            with open(self.file_name, 'w') as file:
                file.write("Temps (s),Puissance (W),Position x (mm),Position y (mm)\n")   
                for i in self.dataArray:
                    file.write(f"{i[0]},{i[1]},{i[3][0]},{i[3][1]}\n")

        elif self.choiceArray == [False, True, True]:
            with open(self.file_name, 'w') as file:
                file.write("Temps (s),Longueur d'onde (nm),Position x (mm),Position y (mm)\n")
                for i in self.dataArray:
                    file.write(f"{i[0]},{i[2]},{i[3][0]},{i[3][1]}\n")
            
        elif self.choiceArray == [True, False, False]:
            with open(self.file_name, 'w') as file:
                file.write("Temps (s),Puissance (W)\n")
                for i in self.dataArray:
                    file.write(f"{i[0]}, {i[1]}\n")
            
        elif self.choiceArray == [False, True, False]:
            with open(self.file_name, 'w') as file:
                file.write("Longueur d'onde (nm)\n")
                for i in self.dataArray:
                    file.write(f"{i[0]},{i[1]}\n")

        elif self.choiceArray == [False, False, True]:
            with open(self.file_name, 'w') as file:
                file.write("Position x (mm),Position y (mm)\n")
                for i in self.dataArray:
                    file.write(f"{i[0]}, {i[3][0]},{i[3][1]}\n")

        elif self.choiceArray == [False, False, False]:
            self.error_handling("Aucune donnée sélectionnée à enregistrer")

        self.savingTime = time.time() - self.savingTimeRef #Calculer le temps d'enregistrement



            
                

        #Réactiver les widgets de l'interface
        if self.choiceArray != [False, False, False]:
            self.after(int(self.savingTime), self.enable_widgets())
            self.sucess_message(f"Données enregistrées avec succès en {self.savingTime:.8f} secondes")

        
    def disable_widgets(self):
        #Désactiver les widgets de l'interface
        self.startButton.configure(state = "disabled")
        self.stopButton.configure(state = "disabled")
        self.resetButton.configure(state = "disabled")
        self.saveButton.configure(state = "disabled")
        self.portComboBox.configure(state = "disabled")
        self.wavelengthCheckButton.configure(state = "disabled") 
        self.powerCheckButton.configure(state = "disabled")
        self.positionCheckButton.configure(state = "disabled")
        self.waveLenghtComboBox.configure(state = "disabled")
        self.tareButton.configure(state= "disabled")
        self.fileName.configure(state = "disabled")

    def enable_widgets(self):
        #Réactiver les widgets de l'interface
        self.startButton.configure(state = "normal")
        self.stopButton.configure(state = "normal")
        self.resetButton.configure(state = "normal")
        self.saveButton.configure(state = "normal")
        self.portComboBox.configure(state = "normal")
        self.wavelengthCheckButton.configure(state = "normal") 
        self.powerCheckButton.configure(state = "normal")
        self.positionCheckButton.configure(state = "normal")
        self.waveLenghtComboBox.configure(state = "normal")
        self.tareButton.configure(state= "normal")
        self.fileName.configure(state = "normal")



    def sucess_message(self, message):
        #Création d'un label pour les messages de succès
        self.errorLabel = ttk.Label(self, text = message, font=("Inter", 20, "bold"), foreground = "green")
        self.errorLabel.place(x = 10, y = 870)

        self.disable_widgets()
        #Afficher le message pendant 3 secondes
        self.errorLabel.after(3000, self.erase_error)

        if self.running == False:
            if len(self.dataArray) != 0:
                self.click_start()
    

    def error_handling(self, message):
        #Création d'un label pour les erreurs
        self.errorLabel = ttk.Label(self, text = message, font=("Inter", 20, "bold"), foreground = "red")
        self.errorLabel.place(x = 10, y = 870)

        #Désactiver les widgets de l'interface
        self.disable_widgets()

        #Afficher l'erreur pendant 3 secondes
        self.errorLabel.after(3000, self.erase_error)
    
    def erase_error(self):
        #Effacer le message d'erreur
        self.errorLabel.destroy()

        #Réactiver les widgets de l'interface
        self.enable_widgets()

    def on_close(self): #Permet de fermer la fenêtre et de fermer le port série

        if self.running == True: #Arrête la simulation si elle est en cours
            self.click_stop()

        if self.serialManager.serialListener is not None: #Déconnecte le port série
            self.serialManager.closePort()
        self.destroy() #Ferme la fenêtre


    def check_ports(self):
        # Start checking for port updates
        if len(self.portList) == 0:

            self.update_ports()
        else:
            check_ports = serial.tools.list_ports.comports()
            for port in check_ports:
                if "Bluetooth" not in port.description:
                    check_ports = [f"{port.device}, {port.description}"]
            if len(check_ports) != len(self.portList):        
                self.update_ports()
            else:
                pass



    def update_ports(self):
        if self.running == True:
            self.click_stop()
        current_ports = serial.tools.list_ports.comports()
        updated_port_list = []
        for port in current_ports:
            if "Bluetooth" not in port.description:
                updated_port_list.append(f"{port.device}, {port.description}")

        if updated_port_list != self.portList:  
            self.portList = updated_port_list
            self.portComboBox['values'] = self.portList
            if len(self.portList) == 1:  
                self.selected_port.set(self.portList[0])
            else:
                self.selected_port.set("")  

        self.after(1000, self.update_ports)  



        
if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.protocol("WM_DELETE_WINDOW", app.on_close)  # Handle window close event
    app.mainloop()