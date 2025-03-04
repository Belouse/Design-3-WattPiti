from mytk import *


# Création de la classe de l'interface graphique pour utilisateur
class InterfaceWattpiti(App):
    def __init__(self):
        App.__init__(self)

        self.window.widget.title("Puissance-mètre Wattpiti")
        self.window.widget.geometry("800x800")
        self.start_button = Button("Commencer", user_event_callback=self.click_start)
        self.start_button.grid_into(self.window, row = 0, column = 0, padx = 500, pady = 400, sticky = "ns")


    def click_start(self, event, button):
        self.start_button.grid_into(self.window, row = 800, column = 800, padx = 0, pady = 0, sticky = "ns")
            


# Lancement de l'interface graphique
if __name__ == "__main__":
    app = InterfaceWattpiti()
    app.mainloop()
