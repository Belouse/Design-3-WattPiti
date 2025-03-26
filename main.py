from MCUv1.JSONFormatterClass import JSONFormatter
from FileManagerClass import FileManager
from AlgorithmManagerClass import AlgorithmManager
from SerialPortClass import SerialPort
from UI import InterfaceWattpiti
from PC.Algo.AlgoPosition import AlgoPosition
from PC.Algo.AlgoPuissance import AlgoPower
from PC.Algo.AlgoLambda import AlgoWavelength
from PC.DataContainerClass import DataContainer
if __name__ == "__main__":
    serial_port = SerialPort()

    data = serial_port.get_data_from_mcu()

    algorithm_manager = AlgorithmManager(data)


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