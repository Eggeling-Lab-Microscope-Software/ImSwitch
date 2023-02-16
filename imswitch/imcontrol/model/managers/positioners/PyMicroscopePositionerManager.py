import importlib
from .PositionerManager import PositionerManager
from microscope.abc import Stage
from imswitch.imcommon.model import pythontools, initLogger

class PyMicroscopePositionerManager(PositionerManager):

    """ Generic PositionerManager for stage handlers supported by the Python Microscope backend.

    Manager properties:

    - ``pyMicroscopeDriver`` -- string describing the Python Microscope
        object to initialize; requires to specify the module
        and the class name, e.g. ``linkam.LinkamCMS``
    """

    def __init__(self, positionerInfo, name: str, **lowLevelManagers):
        self.__logger = initLogger(self)
        self.__driver = str(positionerInfo.managerProperties["pyMicroscopeDriver"])

        axes = (positionerInfo.axes if isinstance(positionerInfo.axes, list) else list(positionerInfo.axes.keys()))

        driver = self.__driver.split(".")
        modulePath = "microscope.stages"
        if driver[0] == "simulators":
            modulePath = "microscope"
        package = importlib.import_module(
            pythontools.joinModulePath(modulePath, driver[0])
        )

        if driver[0] == "simulators":
            from microscope import AxisLimits
            self.__stage : Stage = getattr(package, driver[1])(limits={ax: AxisLimits(-100, 100) for ax in axes})
        else:
            self.__stage : Stage = getattr(package, driver[1])()

        initialPosition = {
            ax: self.__stage.position[ax] for ax in axes
        }

        self.__logger.info(f"{self.__driver} initialized. ")
        super().__init__(positionerInfo, name, initialPosition)
    
    def move(self, dist: float, axis: str):
        self.__stage.move_by({axis: dist})
        self._position[axis] = self.__stage.position[axis]
        
    
    def setPosition(self, position: float, axis: str):
        self.__stage.move_to({axis: position})
        self._position[axis] = self.__stage.position[axis]