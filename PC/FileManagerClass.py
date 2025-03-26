from datetime import datetime

class FileManager():

    def __init__(self, filename=None):

        if filename:
            self.filename = filename
        else: 
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            self.filename = f"data_{dt_string}.csv"

    def save_data(self, data):
        # save data in a file everytime the mcu gives new data points
        pass