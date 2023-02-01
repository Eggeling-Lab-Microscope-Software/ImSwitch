import importlib
from imswitch.imcommon.model import pythontools, initLogger
from microscope.abc import Camera
from microscope import ROI
from collections import deque
from .DetectorManager import (
    DetectorManager, DetectorNumberParameter, DetectorListParameter
)

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
        
        modulePath = "microscope.cameras"
        if driver[0] == "simulators":
            modulePath = "microscope.simulators"
        package = importlib.import_module(
            pythontools.joinModulePath(modulePath, driver[0])
        )
        self.__camera : Camera = getattr(package, driver[1])(index=detectorInfo.managerProperties["cameraIndex"])
        
        # python-microscope cameras work by internally redirecting acquired frames to a queue-like client
        # set by the user; using a built-in deque works, but for higher speed acquisitions it loses a lot of
        # performance; in the future a proper allocated memory section using for example numpy should be used
        self.__client = deque(maxlen=100)
        
        
        # gather dictionary settings and 
        # parse them within the manager parameters
        parameters = {
            "Client buffer size" : DetectorNumberParameter(group="Device client", 
                                                    value=self.__client.maxlen,
                                                    valueUnits="frames",
                                                    editable=True)
        }
        for setting in self.__camera.describe_settings():
            editable = not setting[1]["readonly"]
            settingType = setting[1]["type"]
            if settingType == "enum":
                # enum settings usually contain tuples with
                # [0] being an index of the property and
                # [1] being the name of the property
                # contained values may be also tuples as well,
                # and they need to be converted to string
                options = [str(setting[1]["values"][idx][1]) for idx in range(setting[1]["values"])]
                parameters[f"{setting[0]} (enum)"] = DetectorListParameter(group="Camera settings", 
                                                                value=str(options[0]),
                                                                options=options,
                                                                editable=editable)
            elif settingType == "int":
                limits = setting[1]["values"] if editable else (None, None)
                # some limit values may be on the range [-X, X];
                # better make sure that the starting value is set
                # in the middle range if this happens
                startValue = limits[0]
                if -limits[0] == limits[1]:
                    startValue = 0
                parameters[f"{setting[0]} (int)"] = DetectorNumberParameter(group="Camera settings",
                                                                            value=startValue,
                                                                            editable=editable,
                                                                            valueUnits="",
                                                                            valueLimits=limits)
            elif settingType == "str":
                # we don't know if string parameters are always readonly,
                # so we cast them to a list and treat them as list parameters
                options = [setting[1]["values"]]
                parameters[f"{setting[0]} (str)"] = DetectorListParameter(group="Camera settings",
                                                                          value=options[0],
                                                                          options=options,
                                                                          editable=editable)
            elif settingType == "bool":
                # using 0 and 1 is not very comfortable to use for bool values,
                # so we treat them as strings "OFF" and "ON"
                options = ["OFF", "ON"]
                parameters[f"{setting[0]} (bool)"] = DetectorListParameter(group="Camera settigs",
                                                                           value=options[0],
                                                                           options=options,
                                                                           editable=editable)
            else:
                raise ValueError(f"Setting type {settingType} is unrecognized!")
        
        self.__logger.info(f"{self.__driver} initialized.")
        super().__init__(detectorInfo, name)
    
    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def setParameter(self, name, value):
        if name == "Client buffer size":
            self.__client.clear()
            self.__client = deque(maxlen=value)
        else:
            if "(bool)" in name:
                value = 0 if value == "OFF" else "1"
            self.__camera.set_setting(name, value)
        return super().setParameter(name, value)
    
    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        """ Crop the frame read out by the detector. """
        newROI = ROI(hpos, vpos, hsize, vsize)
        if not self.__camera.set_roi(newROI):
            self.__logger.error(f"Failed to set ROI {newROI}")
            

    def getLatestFrame(self, is_save=False):
        """ Returns the frame that represents what the detector currently is
        capturing. The returned object is a numpy array of shape
        (height, width). """
        self.__camera

    def getChunk(self):
        """ Returns the frames captured by the detector since getChunk was last
        called, or since the buffers were last flushed (whichever happened
        last). The returned object is a numpy array of shape
        (numFrames, height, width). """
        pass

    def flushBuffers(self) -> None:
        """ Flushes the detector buffers so that getChunk starts at the last
        frame captured at the time that this function was called. """
        pass

    def startAcquisition(self) -> None:
        """ Starts image acquisition. """
        pass

    def stopAcquisition(self) -> None:
        """ Stops image acquisition. """
        pass

    def finalize(self) -> None:
        """ Close/cleanup detector. """
        pass