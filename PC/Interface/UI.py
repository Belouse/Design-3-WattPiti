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
        self.configure(background="grey")


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

        #self.start_button = ttk.Button(self, text="Commencer", command=self.click_start)
        #self.start_button.grid(row=5, column=5, padx=5, pady=5, sticky="nsew")

    def click_start(self):
        pass

if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()







"""
from mytk import *
import os
# Création de la classe de l'interface graphique pour utilisateur
class InterfaceWattpiti(App):
    def __init__(self):
        App.__init__(self)

        self.window.widget.title("Puissance-mètre Wattpiti")
        self.window.widget.geometry("1800x1800")

        self.window.row_resize_weight(0,0)
        self.window.row_resize_weight(1,0)
        self.window.row_resize_weight(2,1)
        self.window.column_resize_weight(0,1)
        self.window.column_resize_weight(1,1)


        self.view = View(width=1000, height=1000)
        self.view.grid_into(self.window, row =0, column = 0, pady=5, padx=5, sticky="nsew")
        self.view.grid_propagate(True)
        dir = os.path.dirname(__file__)
        self.logo = Image(os.path.join(dir,"logoWattpiti.jpg"))
        self.logo.grid_into(self.view, row = 1, column = 0, padx = 5, pady = 5, sticky ="")



        #self.start_button = Button("commencer",  user_event_callback= self.click_start)
        #self.start_button.grid_into(self.window, row = 0, column = 0, padx = 100, pady = 100, sticky = "nsew")


    def click_start(self, button, event):
        pass


# Lancement de l'interface graphique

if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()
"""