from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.interfaces.MadCityLabs import (
    MicroDriveHandler,
    NanoDriveHandler
)
from .PositionerManager import PositionerManager

class MCLPositionerManager(PositionerManager):
    """ PositionerManager for Mad City Labs drivers.

    Manager properties:
    - ``type`` -- string value: "MicroDrive" or "NanoDrive" supported
    - ``dllPath`` -- string containing the absolute path to the DLL
    """

    def __init__(self, positionerInfo, name, **lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.__driver = None
        posType = positionerInfo.managerProperties["type"]

        if posType == "MicroDrive":
            self.__driver = MicroDriveHandler(positionerInfo.managerProperties["dllPath"], positionerInfo.axes, self.__logger)
        elif posType == "NanoDrive":
            self.__driver = NanoDriveHandler(positionerInfo.managerProperties["dllPath"], positionerInfo.axes, self.__logger)
            pass
        else:
            raise RuntimeError(f"Unsupported positioner type {posType}")

        super().__init__(positionerInfo, name, initialPosition={
            axis : 0.0 for axis in positionerInfo.axes.keys()
        })

        for info in self.__driver.getDriverInfo():
            self.__logger.info(info)

    def move(self, dist: float, axis: str):
        self.__driver.setPosition(axis, dist)
        self.position[axis] += dist
    
    def setPosition(self, position: float, axis: str):
        self.__driver.setPosition(axis, position)
        self.position[axis] = position
    
    def calibrate(self) -> bool:
        # todo: fix
        # if self.__driver.calibrate():
        #     # calibration successful
        #     for axis in self.axes:
        #         self.position[axis] = 0.0
        #     return True
        # else:
        #     return False
        return False
    
    def finalize(self) -> None:
        self.__driver.close()