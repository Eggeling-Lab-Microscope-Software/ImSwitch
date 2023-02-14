import importlib
import numpy as np
from dataclasses import dataclass
from imswitch.imcommon.model import pythontools, initLogger
from microscope.abc import Camera
from microscope import ROI, TriggerMode, TriggerType, UnsupportedFeatureError
from queue import Queue
from .DetectorManager import (
    DetectorManager, DetectorNumberParameter, DetectorListParameter
)

@dataclass
class PyMicroscopeSetting(DetectorNumberParameter):
    valueLimits: tuple = (None, None)
    """ Parameter value upper and lower limits. """


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
            modulePath = "microscope"
        package = importlib.import_module(
            pythontools.joinModulePath(modulePath, driver[0])
        )
        self.__camera : Camera = getattr(package, driver[1])(index=detectorInfo.managerProperties["cameraIndex"])
        fullShape = self.__camera.get_sensor_shape()
        self.__client = Queue(maxsize=100)
        
        # python-microscope cameras work by internally redirecting acquired frames to a queue-like client
        # set by the user; using a built-in deque works, but for higher speed acquisitions it loses a lot of
        # performance; in the future a proper allocated memory section using for example numpy should be used        
        
        # gather dictionary settings and 
        # parse them within the manager parameters
        parameters = {
            "Client buffer size" : DetectorNumberParameter(group="Device client", 
                                                    value=self.__client.maxsize,
                                                    valueUnits="frames",
                                                    editable=True)
        }
        parameters["Trigger type"] = DetectorListParameter(group="Trigger settings",
                                                        value=TriggerType.SOFTWARE.name,
                                                        options=[type.name for type in TriggerType],
                                                        editable=True)
        parameters["Trigger mode"] = DetectorListParameter(group="Trigger settings",
                                                           value=TriggerMode.ONCE.name,
                                                           options=[mode.name for mode in TriggerMode],
                                                           editable=True)
        parameters["Exposure"] = DetectorNumberParameter(group="Timings",
                                                        value=1e-3,
                                                        valueUnits="s",
                                                        editable=True)
        for setting in self.__camera.describe_settings():
            editable = not setting[1]["readonly"]
            settingType = setting[1]["type"]
            if settingType == "enum":
                paramKey = setting[0]
                if paramKey != "transform":
                    paramKey += " (enum)"

                # enum settings usually contain tuples with
                # [0] being an index of the property and
                # [1] being the name of the property
                # contained values may be also tuples as well,
                # and they need to be converted to string
                options = [str(setting[1]["values"][idx]) for idx in range(len(setting[1]["values"]))]

                # this is just some string formatting
                options = [option[1:len(option)-1] for option in options]
                parameters[paramKey] = DetectorListParameter(group="Camera settings", 
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
                # ROI and transform are special settings which are handled by differently;
                # the ROI is set via "crop", the transform is indexed rather than 
                if setting[0] != "roi":
                    raise ValueError(f"Setting type {settingType} is unrecognized!")
        
        parameters["Camera pixel size"] = DetectorNumberParameter(group="Miscellaneous",
                                                                value=10,
                                                                valueUnits="µm",
                                                                editable=True)

        self.__logger.info(f"{self.__driver} initialized.")

        # TODO: binning requires clarification;
        # how do we know which binnings are available
        # depending on the device?
        # binnings = self.__camera.get_binning()
        super().__init__(detectorInfo, name, fullShape, supportedBinnings=[1], model=driver[1], parameters=parameters, croppable=True)
        self.__camera.set_client(self.__client)
    
    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def setParameter(self, name, value):
        if name == "Client buffer size":
            self.__client.queue.clear()
            self.__client = Queue(maxsize=int(value))
        elif name == "Trigger type":
            try:
                self.__camera.set_trigger(TriggerType[value], self.__camera.trigger_mode)
            except UnsupportedFeatureError:
                self.__logger.error(f"{value} is not a supported TriggerType for this device")
        elif name == "Trigger mode":
            try:
                self.__camera.set_trigger(self.__camera.trigger_type, TriggerMode[value])
            except UnsupportedFeatureError:
                self.__logger.error(f"{value} is not a supported TriggerMode for this device")
        elif name == "Exposure":
            self.__camera.set_exposure_time(float(value))
        elif name == "transform":
            value = value.split()
            self.__logger.info(value)
            self.__camera.set_transform(value)
        else:
            if "(bool)" in name:
                value = 0 if value == "OFF" else 1
            elif "(enum)" in name:
                value = int(value.replace(",", "").split()[0])
            # we have to remove the last part of the parameter name,
            # as it is incompatible with microscope
            settingName = name.split()
            settingName = " ".join(settingName[0: len(settingName)-1])
            self.__camera.set_setting(settingName, value)
        return super().setParameter(name, value)
    
    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        """ Crop the frame read out by the detector. """
        newROI = ROI(hpos, vpos, hsize, vsize)
        self.__camera.set_roi(newROI)
        if self.__camera.get_roi() != newROI:
            self.__logger.error(f"Failed to set ROI {newROI}")
            raise ValueError()
        self._frameStart = (hpos, vpos)
        self._shape = (hsize, vsize)
            

    def getLatestFrame(self, is_save=False):
        """ Returns the frame that represents what the detector currently is
        capturing. The returned object is a numpy array of shape
        (height, width). """
        
        if self.__camera.trigger_type == TriggerType.SOFTWARE:
            self.__camera.trigger()
        return self.__client.get()
        

    def getChunk(self):
        """ Returns the frames captured by the detector since getChunk was last
        called, or since the buffers were last flushed (whichever happened
        last). The returned object is a numpy array of shape
        (numFrames, height, width). """
        return np.stack(self.__client)

    def flushBuffers(self) -> None:
        """ Flushes the detector buffers so that getChunk starts at the last
        frame captured at the time that this function was called. """
        self.__client.queue.clear()

    def startAcquisition(self) -> None:
        """ Starts image acquisition. """
        self.__camera.enable()

    def stopAcquisition(self) -> None:
        """ Stops image acquisition. """
        self.__camera.disable()

    def finalize(self) -> None:
        """ Close/cleanup detector. """
        self.stopAcquisition()