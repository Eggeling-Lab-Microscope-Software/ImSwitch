from pprint import pprint
from .PositionerManager import PositionerManager
from ..PyMMCoreManager import PyMMCoreManager # only for type hinting
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.SetupInfo import SetupInfo
from imswitch.imcontrol.model.configfiletools import (
    loadOptions, 
    loadSetupInfo,
    saveSetupInfo
)

class PyMMCorePositionerManager(PositionerManager):
    """ PositionerManager for control of a stage controlled by the Micro-Manager core, using pymmcore.

    Manager properties:
    - ``module`` -- name of the MM module referenced
    - ``device`` -- name of the MM device described in the module 
    - ``stageType`` -- either "single" or "double" (for single-axis stage or double-axis stage)
    - ``speedProperty`` (optional) -- name of the property indicating the stage speed
    """

    def __init__(self, positionerInfo, name: str, **lowLevelManagers):

        self.__logger = initLogger(self)
        self.__stageType = positionerInfo.managerProperties["stageType"]
        # only combination of double axis stages allowed is X-Y
        if self.__stageType == "double":
            if len(positionerInfo.axes) != 2:
                raise ValueError(f"Declared axis number not correct. Must be 2 ([\"X\", \"Y\"]), instead is {len(positionerInfo.axes)}")
            elif positionerInfo.axes != ["X", "Y"]:
                raise ValueError(f"Declared axis names incorrect. Must be [\"X\", \"Y\"], instead is {positionerInfo.axes}")

        # type assignment useful for type hinting
        self.__coreManager: PyMMCoreManager = lowLevelManagers["pymmcManager"]

        module = positionerInfo.managerProperties["module"]
        device = positionerInfo.managerProperties["device"]
        try:
            self.__speedProp = positionerInfo.managerProperties["speedProperty"]
        except:
            self.__speedProp = None

        self.__logger.info(f"Loading {name}.{module}.{device} ...")

        devInfo = (name, module, device)
        self.__coreManager.loadDevice(devInfo)

        # can be read only after device is loaded and initialized
        # some device may not have a speed property...
        if self.__speedProp is not None:
            self.speed =  float(self.__coreManager.getProperty(name, self.__speedProp))
        else:
            # assuming device has no speed
            self.speed = 0.0
        self.__logger.info(f"... done!")
    
        assert len(positionerInfo.axes) == len(positionerInfo.startPositions), "Axes and starting positions don't match length."
        assert positionerInfo.axes == list(positionerInfo.startPositions.keys()), "Axes and starting positions don't match names."
        
        initialPosition = {
            axis : positionerInfo.startPositions[axis] for axis in positionerInfo.axes 
        }         
        super().__init__(positionerInfo, name, initialPosition)
    
    def setPosition(self, position: float, axis: str) -> None:
        try:
            oldPosition = self.position[axis]
            self._position[axis] = position
            self._position = self.__coreManager.setStagePosition(
                self.name,
                self.__stageType,
                axis,
                self.position
            )
        except RuntimeError:
            self.__logger.error(f"Invalid position requested ({self.name} -> ({axis} : {position})")
            self._position[axis] = oldPosition
    
    def setSpeed(self, speed: float) -> None:
        # check if speed property exists
        if self.__speedProp is not None:
            self.__coreManager.setProperty(self.name, self.__speedProp, speed)
    
    def move(self, dist: float, axis: str) -> None:
        movement = {ax : 0.0 for ax in self.axes}
        movement[axis] = dist
        try:
            self.__coreManager.moveStage(
                self.name,
                self.__stageType,
                axis,
                movement
            )
            self._position[axis] += dist
        except RuntimeError:
            self.__logger.error(f"Invalid movement requested ({self.name} -> ({axis} : {dist})")
    
    def finalize(self) -> None:
        if self.storePosition:
            self.__logger.info("Storing current position in setup file...")
            options, optionsDidNotExist = loadOptions()
            if not optionsDidNotExist:
                # options found
                setupInfo = loadSetupInfo(options, SetupInfo)
                setupInfo.positioners[self.name].startPositions = self.position
                saveSetupInfo(options, setupInfo)
                self.__logger.info("... done!")
            else:
                # options not found
                self.__logger.error("... could not find setup file! Skipping.")
        self.__logger.info(f"Closing {self.name}.")
        self.__coreManager.unloadDevice(self.name)