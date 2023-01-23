import importlib
from .DetectorManager import DetectorManager
from imswitch.imcommon.model import pythontools, initLogger
from microscope.abc import Camera

class PyMicroscopeCameraManager(DetectorManager):
    """Generic DetectorManager for camera handlers supported by the Python Microscope backend.

    Manager properties:

    - ``pyMicroscopeDriver`` -- string describing the Python Microscope
        object to initialize; requires to specify the module
        and the class name, e.g. ``pvcam.PVCamera``
    - ``cameraIndex`` -- integer specifying the index of the device in a shared library.
    """
    def __init__(self, detectorInfo, name, **_lowLevelManager) -> None:
        self.__logger = initLogger(self, instanceName=name)
        self.__driver = str(detectorInfo.managerProperties["pyMicroscopeDriver"])
        driver = self.__driver.split(".")
        package = importlib.import_module(
            pythontools.joinModulePath("microscope.cameras", driver[0])
        )
        self.__camera : Camera = getattr(package, driver[1])(index=detectorInfo.managerProperties["cameraIndex"])

        # gather dictionary settings and 
        # parse them within the manager parameters
        parameters = {}
        settings = self.__camera.get_all_settings()

        for name, setting in settings.items():
            parameters[name] = setting

        self.__logger.info(f"[{self.__port}] {self.__driver} initialized. ")
        super().__init__(detectorInfo, name)
    
    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]
