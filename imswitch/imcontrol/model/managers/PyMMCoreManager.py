from imswitch.imcommon.framework import SignalInterface
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.configfiletools import _mmcoreLogDir
from typing import Union, Tuple, Dict, List
import datetime as dt
import os
import pymmcore

PropertyValue = Union[bool, float, int, str]

class PyMMCoreManager(SignalInterface):
    """ For interaction with Micro-Manager C++ core. 
        Using pymmcore package (a Python-API wrapper).

        Setup fields:
        - ``MMPath``: MM absolute path in the system
        - ``MMDevSearchPath`` (optional): list of MM device search paths. If not set, ``MMPath`` is taken as reference.
    """

    def __init__(self, setupInfo) -> None:
        super().__init__()
        self.__logger = initLogger(self)
        mmPath = setupInfo.pymmcore.MMPath

        if mmPath is None:
            raise ValueError("No Micro-Manager path defined.")

        self.__core = pymmcore.CMMCore()
        devSearchPath = (setupInfo.pymmcore.MMDevSearchPath 
                        if setupInfo.pymmcore.MMDevSearchPath is not None 
                        else mmPath)

        if not isinstance(devSearchPath, list):
            devSearchPath = [devSearchPath]
        
        self.__logger.info(f"Micro-Manager path: {mmPath}")
        self.__logger.info(f"Device search paths: {devSearchPath}")

        self.__core.setDeviceAdapterSearchPaths(devSearchPath)
        self.__logger.info(self.__core.getAPIVersionInfo())

        self.__getXYStagePosition = {
            "X" : self.__core.getXPosition,
            "Y" : self.__core.getYPosition
        }
        
        logpath = os.path.join(_mmcoreLogDir, dt.datetime.now().strftime("%d_%m_%Y") + ".log")
        self.__core.setPrimaryLogFile(logpath)
        self.__core.enableDebugLog(True)
    
    def loadDevice(self, devInfo: Tuple[str, str, str]) -> None:
        """ Tries to load a device into the MMCore.

        Args:
            devInfo (``tuple[str, str, str]``): a tuple describing the device information. It's arranged as:
            - devInfo[0]: label
            - devInfo[1]: moduleName
            - devInfo[2]: deviceName
        """
        try:
            self.__core.loadDevice(
                devInfo[0],
                devInfo[1],
                devInfo[2]
            )
            self.__core.initializeDevice(devInfo[0])
        except Exception as e:
            raise ValueError(f"Error in loading device \"{devInfo[0]}\", check the values of \"module\" and \"device\" in the configuration file (current values: {devInfo[1]}, {devInfo[2]}) ({e})")
    
    def unloadDevice(self, label: str) -> None:
        """ Tries to unload from the MMCore a previously loaded device (used for finalize() call)
        """
        try:
            self.__core.unloadDevice(label)
        except RuntimeError:
            raise ValueError(f"Error in unloading device \"{label}\"")
    
    def getProperty(self, label: str, property: str) -> str:
        """ Returns the property of a device.

        Args:
            label (``str``): name of the device
            property (``str``): label of the property to read            
        """
        try:
            return self.__core.getProperty(label, property)
        except Exception as err:
            raise RuntimeError(f"Failed to load property \"{property}\": {err.__str__()}")
    
    def setProperty(self, label: str, property: str, value: PropertyValue) -> None:
        """ Sets the property of a device.
        
        Args:
            label (``str``): name of the device
            property (``str``): label of the property to read
            value (``PropertyValue``): value to set the property with
        """
        try:
            self.__core.setProperty(label, property, value)
        except RuntimeError as err:
            self.__logger.error(f"Failed to set \"{property}\" to {value}: {err.__str__()}")
    
    def getStagePosition(self, label: str, axis: str) -> float:
        """ Returns the current stage position (on a given axis for double-axis stages).
        
        Args:
            label (``str``): name of the positioner
            axis (``str``): axis to read
        """
        return (self.__getXYStagePosition[axis](label) if axis in self.__getXYStagePosition.keys() else self.__core.getPosition(label))
    
    def setStagePosition(self, label: str, stageType: str, axis: str, positions: Dict[str, float], isAbsolute: bool = True) -> Dict[str, float]:
        """ Sets the selected stage position.

        Args:
            label (``str``): name of the positioner
            stageType (``str``): type of positioner (either "single" or "double")
            axis (``str``): axis to move (used only for "single" stage type)
            positions (``dict[str, float]``): dictionary with the positions to set.
            isAbsolute (``bool``): ``True`` if absolute movement is requested, otherwise false.  
        
        Returns:
            the dictionary with the new [axis, position] assignment.
        """
        if stageType == "single":
            if isAbsolute:
                self.__core.setPosition(label, positions[axis])
            else:
                self.__core.setRelativePosition(label, positions[axis])
            positions[axis] = self.getStagePosition(label, axis)
        else:
            # axis are forced by the manager constructor
            # to be "X-Y", so this call should be safe
            # just keep it under control...
            if isAbsolute:
                self.__core.setXYPosition(label, positions["X"], positions["Y"]) 
            else:
                self.__core.setRelativeXYPosition(label, positions["X"], positions["Y"])
            positions = {axis : self.__getXYStagePosition[axis](label) for axis in ["X", "Y"]}
        return positions
    
    def setStageOrigin(self, label: str, stageType: str, axes: List[str]) -> Dict[str, float]:
        """Zeroes the stage at the current position.

        Args:
            label (``str``): name od the positioner
            stageType (``str``): type of positioner (either "single" or "double")
            axis (str): axis to move (used only for "single" stage type)

        Returns:
            Dict[str, float]: dictionary containing the new positioner's origin.
        """
        positions = {}
        if stageType == "single":
            self.__core.setOrigin(label)
            positions[axes[0]] = self.__core.getPosition(label)
        else:
            self.__core.setOriginXY(label)
            positions = {ax : self.__getXYStagePosition[ax](label) for ax in axes}
        return positions
        
