from mytk import *
import time
from tkinter import ttk
from tkinter import *


# Création de la classe de l'interface graphique pour utilisateur
class InterfaceWattpiti(App):
    def __init__(self):
        App.__init__(self)

        self.window.widget.title("Puissance-mètre Wattpiti") # Nom de la fenêtre
        self.window.widget.geometry("400x400")
        self.progress = ttk.Progressbar(self.window.widget, length=200)
        self.progress.pack(pady=20)
        
        self.start_button = Button(self.window.widget, text = "Start",  command = self.start_progress)
        self.start_button.pack(pady=10)
        self.stop_button = Button(self.window.widget, text = "Stop", command = self.stop_progress)
        self.stop_button.pack(pady=10)
        self.stop = False

    def start_progress(self):
        self.start_button.pack_forget()
        self.progress.pack_forget()
        self.stop_button.pack_forget()
        self.progress['value'] = 0
        self.window.widget.update_idletasks()
        for i in range(101):
            if self.stop:
                break
            time.sleep(0.05)
            self.progress['value'] = i
            self.window.widget.update_idletasks()
        self.start_button.pack(pady=10)
        self.progress.pack(pady=20)
        self.stop_button.pack(pady=10)

    def stop_progress(self):
        self.stop = True


# Lancement de l'interface graphique
if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()
