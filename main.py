from JSONFormatterClass import JSONFormatter
from FileManagerClass import FileManager
from AlgorithmManagerClass import AlgorithmManager
from SerialPortClass import SerialPort
from UI import InterfaceWattpiti


if __name__ == "__main__":
    serial_port = SerialPort()

    data = serial_port.get_data_from_mcu()

    algorithm_manager = AlgorithmManager(data)

    file_manager = FileManager()
    file_manager.save_data(data)

    position = algorithm_manager.calculate_position()
    power = algorithm_manager.calculate_power()
    wavelength = algorithm_manager.calculate_wavelength()


    interface = InterfaceWattpiti()
    interface.set_position(position)
    interface.set_power(power)
    interface.set_wavelength(wavelength)
    interface.mainloop()

    print(position)
    print(power)
    print(wavelength)