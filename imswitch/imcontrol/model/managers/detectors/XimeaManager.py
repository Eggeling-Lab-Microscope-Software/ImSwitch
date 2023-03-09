import numpy as np
import Pyro5.api
import re, os
from tifffile.tifffile import imwrite
from ximea.xiapi import Xi_error
from imswitch.imcontrol.model.interfaces import XimeaSettings
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model.dirtools import UserFileDirs
from imswitch.imcommon.framework.qt import Timer
from contextlib import contextmanager
from numba import vectorize, float32
from .DetectorManager import (
    DetectorManager, DetectorNumberParameter, DetectorListParameter, DetectorAction
)
from qtpy.QtWidgets import QFileDialog

@vectorize([float32(float32, float32)], cache=True, nopython=True)
def numba_matrix_division(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x / y

@vectorize([float32(float32, float32)], cache=True, nopython=True)
def numba_matrix_subtraction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y

class XimeaManager(DetectorManager):
    """ DetectorManager that deals with the Ximea parameters and frame
    extraction for a Ximea camera.

    Manager properties:

    - ``cameraListIndex`` -- the camera's index in the Ximea camera list
      (list indexing starts at 0); set this to an invalid value, e.g. the
      string "mock" to load a mocker
    - ``parameters`` -- dictionary of XiAPI properties to pass to the driver
    - ``medianFilter`` -- dictionary of parameters required for median filter acquisition
    - ``server`` -- URI of local ImSwitchServer in the format of host-port (i.e. \"127.0.0.1:54333\") 
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self._camera, self._img = self._getCameraObj(detectorInfo.managerProperties['cameraListIndex'])
        self._median : np.ndarray = None

        self._settings = XimeaSettings(self._camera)

        # open Ximea camera for allowing parameters settings
        self._camera.open_device()

        fullShape = (self._camera.get_width_maximum(),
                     self._camera.get_height_maximum())

        model = self._camera.get_device_info_string("device_name").decode("utf-8")

        for propertyName, propertyValue in detectorInfo.managerProperties['parameters'].items():
            self._camera.set_param(propertyName, propertyValue)
        
        server = detectorInfo.managerProperties["server"]
        self.__uri = None

        if server is None:
            self.__uri = None
        elif server == "default":
            self.__uri = Pyro5.api.URI("PYRO:ImSwitchServer@127.0.0.1:54333")
        else:
            uri_str = "PYRO:ImSwitchServer@" + server
            self.__uri = Pyro5.api.URI(uri_str)

        # gather parameters for median filter control
        try:
            self.__mfPositioners = detectorInfo.managerProperties["medianFilter"]["positioners"]
            self.__mfStep = detectorInfo.managerProperties["medianFilter"]["step"]
            self.__mfMaxFrames = detectorInfo.managerProperties["medianFilter"]["maxFrames"]
        except:
            self.__logger.warning("No information available for median filter control.")
            self.__mfPositioners = None
            self.__mfStep = None
            self.__mfMaxFrames = None

        # prepare parameters
        parameters = {
            'Exposure': DetectorNumberParameter(group='Timings', value=100e-6,
                                                valueUnits='s', editable=True),
            'Frame rate': DetectorNumberParameter(group='Timings', value=0,
                                                valueUnits='FPS', editable=True),
            
            'Trigger source': DetectorListParameter(group='Trigger settings',
                                                    value=list(self._settings.settings[0].keys())[0],
                                                    options=list(self._settings.settings[0].keys()),
                                                    editable=True),
            
            'Trigger type': DetectorListParameter(group='Trigger settings',
                                                value=list(self._settings.settings[1].keys())[0],
                                                options=list(self._settings.settings[1].keys()),
                                                editable=True),
            
            'GPI': DetectorListParameter(group='Trigger settings',
                                        value=list(self._settings.settings[2].keys())[0],
                                        options=list(self._settings.settings[2].keys()),
                                        editable=True),

            'GPI mode': DetectorListParameter(group='Trigger settings',
                                            value=list(self._settings.settings[3].keys())[0],
                                            options=list(self._settings.settings[3].keys()),
                                            editable=True),
            
            'Bit depth' : DetectorListParameter(group='Miscellaneous',
                                                value=list(self._settings.settings[4].keys())[0],
                                                options=list(self._settings.settings[4].keys()),
                                                editable=True),

            'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=13.7,
                                                         valueUnits='µm', editable=True),                                                        
        }

        actions = {}
        if (self.__uri is not None
                and self.__mfPositioners is not None
                and self.__mfStep is not None
                and self.__mfMaxFrames is not None):
            
            parameters["Median filter step size"] = DetectorNumberParameter(group="Median Filter", value=self.__mfStep, valueUnits="µm", editable=True)
            parameters["Median filter stack size"] = DetectorNumberParameter(group="Median Filter", value=self.__mfMaxFrames, valueUnits="", editable=True)

            actions["Generate median filter"] = DetectorAction(group="Median Filter", func=self._generateMedianFilter)
            parameters["Median filter operation"] = DetectorListParameter(group="Median Filter", value="Division", options=["Division", "Subtraction"], editable=True)
            actions["Clear median filter"] = DetectorAction(group="Median Filter", func=self._clearMedianFilter)
            actions["Store median filter"] = DetectorAction(group="Median Filter", func=self._storeMedianFilter)
            self.__medianFilterOp = numba_matrix_division

        super().__init__(detectorInfo, name, fullShape=fullShape, supportedBinnings=[1],
                         model=model, parameters=parameters, croppable=True, actions=actions)

        self._frameInterval = parameters["Exposure"].value * 1e6
        
        # apparently the XiAPI for detecting if camera is in acquisition does not work
        # we need to use a flag
        self._isAcquiring = False
        self._prevFrameNum = 0
        self._prevFrameTimestamp = 0
        self._newFrameNum = 1
        self._newFrameTimestamp = 1e6
        
        self._fpsTimer = Timer()
        self._fpsTimer.setInterval(250)
        self._fpsTimer.timeout.connect(self._updateFPS)
    
    def _updateFPS(self):
        fps = (self._newFrameNum - self._prevFrameNum) / ((self._newFrameTimestamp - self._prevFrameTimestamp)*1e-6)
        self._prevFrameNum = self._newFrameNum
        self._prevFrameTimestamp = self._newFrameTimestamp
        if fps > 0:
            self.setParameter("Frame rate", fps)
        
    
    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def getLatestFrame(self, is_save=False):
        self._camera.get_image(self._img)
        self._newFrameNum = self._img.nframe
        self._newFrameTimestamp = self._img.tsUSec
        data = self._img.get_image_data_numpy()    
        # median filter applied only if exists in dictionary and its enabled
        if "medianFilter" in self.imageProcessing:
            data = self.__medianFilterOp(data.astype(np.float32), self.imageProcessing["medianFilter"]["content"].astype(np.float32))
        return data

    def getChunk(self):
        return np.stack([self.getLatestFrame()])

    def flushBuffers(self):
        pass
    
    @contextmanager
    def _camera_disabled(self):
        if self._isAcquiring:
            try:
                self.stopAcquisition()
                yield
            finally:
                self.startAcquisition()
        else:
            yield
    
    @contextmanager
    def _camera_enabled(self):
        if not self._isAcquiring:
            try:
                self.startAcquisition()
                yield
            finally:
                self.stopAcquisition()
        else:
            yield

    def crop(self, hpos, vpos, hsize, vsize):
        """Method to crop the frame read out by the camera. """

        # Ximea ROI (at least for xiB-64) works only if the increment is performed
        # using a multiple of the minimum allowed increment of the sensor.
        with self._camera_disabled():
            if (hsize, vsize) != self.fullShape:
                try:
                    self.__logger.debug(f"Crop requested: X0 = {hpos}, Y0 = {vpos}, width = {hsize}, height = {vsize}")
                    hsize_incr = self._camera.get_width_increment()
                    vsize_incr = self._camera.get_height_increment()
                    hsize = (round(hsize / hsize_incr)*hsize_incr if (hsize % hsize_incr) != 0 else hsize)
                    vsize = (round(vsize / vsize_incr)*vsize_incr if (vsize % vsize_incr) != 0 else vsize)
                    self._camera.set_width(hsize)
                    self._camera.set_height(vsize)

                    hpos_incr  = self._camera.get_offsetX_increment()
                    vpos_incr  = self._camera.get_offsetY_increment()
                    hpos = (round(hpos / hpos_incr)*hpos_incr if (hpos % hpos_incr) != 0 else hpos)
                    vpos = (round(vpos / vpos_incr)*vpos_incr if (vpos % vpos_incr) != 0 else vpos)
                    self._camera.set_offsetX(hpos)
                    self._camera.set_offsetY(vpos)                
                    
                    self.__logger.debug(f"Increment info: X0_incr = {hpos_incr}, Y0_incr = {vpos_incr}, width_incr = {hsize_incr}, height_inc = {vsize_incr}")
                    self.__logger.debug(f"Actual crop: X0 = {hpos}, Y0 = {vpos}, width = {hsize}, height = {vsize}")
                except Xi_error as error:
                    self.__logger.error(f"Error in setting ROI (X0 = {hpos}, Y0 = {vpos}, width = {hsize}, height = {vsize})")
                    self.__logger.error(error)
            else:
                self._camera.set_offsetX(0)
                self._camera.set_offsetY(0)
                self._camera.set_width(self.fullShape[0])
                self._camera.set_height(self.fullShape[1])
                
        # This should be the only place where self.frameStart is changed
        self._frameStart = (hpos, vpos)
        # Only place self.shapes is changed
        self._shape = (hsize, vsize)

    def setBinning(self, binning):
        # todo: Ximea binning works in a different way
        # investigate how to properly update this value
        # if possible
        super().setBinning(binning)

    def setParameter(self, name : str, value):

        # this is horrible, but to handle this better
        # we are forced to use Python 3.10...
        if(name in ["Median filter step size", 
                    "Median filter stack size", 
                    "Median filter operation", 
                    "Frame rate"]):
            if name == "Median filter step size":
                self.__mfStep = value
            elif name == "Median filter stack size":
                self.__mfMaxFrames = value
            elif name == "Median filter operation":
                if value == "Division":
                    self.__medianFilterOp = numba_matrix_division
                elif value == "Subtraction":
                    self.__medianFilterOp = numba_matrix_subtraction
            else:
                # nothing to do
                pass
            super().setParameter(name, value)
            return self.parameters

        # Ximea parameters should follow the naming convention
        # described in https://www.ximea.com/support/wiki/apis/XiAPI_Manual
        ximea_value = None

        if name == "Exposure":
            # value must be translated into microseconds
            ximea_value = int(value*1e6)
            self._frameInterval = float(ximea_value)
        else:
            for setting in self._settings.settings:
                if value in setting.keys():
                    ximea_value = setting[value]
                    break
        try:
            self.__logger.info(f"Setting {name} to {ximea_value}")
            if name == "Bit depth":
                with self._camera_disabled():    
                    self._settings.set_parameter[name](ximea_value)
            else:
                self._settings.set_parameter[name](ximea_value)
            # update local parameters
            super().setParameter(name, value)
        except:
            self.__logger.error(f"Cannot set {name} to {ximea_value}")

        return self.parameters

    def startAcquisition(self):
        self._isAcquiring = True
        self._camera.start_acquisition()
        self._fpsTimer.start()

    def stopAcquisition(self):
        self._isAcquiring = False
        self._fpsTimer.stop()
        self._camera.stop_acquisition()
    
    def finalize(self) -> None:
        self._camera.close_device()

    def _getCameraObj(self, cameraId):

        try:
            from ximea.xiapi import Camera, Image
            self.__logger.debug(f'Trying to initialize Ximea camera {cameraId}')
            camera = Camera()
            image = Image()
            
            camera_name = camera.get_device_info_string("device_name").decode("utf-8")
        except:
            self.__logger.warning(f'Failed to initialize Ximea camera {cameraId},'
                                  f' loading mocker')
            from imswitch.imcontrol.model.interfaces import MockXimea, MockImage
            camera = MockXimea()
            image = MockImage()
            camera_name = camera.get_device_info_string("device_name")

        self.__logger.info(f'Initialized camera, model: {camera_name}')
        return camera, image

    def _generateMedianFilter(self):
        if self.__uri is not None:
            try:
                with Pyro5.api.Proxy(self.__uri) as proxy:
                    proxy.generateMedianFilter(self.name, self.__mfPositioners, self.__mfStep, self.__mfMaxFrames)
                    self._dtype = "float32"
            except Exception as e:
                self.__logger.error(f"Could not connect proxy to ImSwitchServer. Error: {e}")
    
    def _clearMedianFilter(self):
        self.__logger.info("Clearing median filter")
        self.imageProcessing.pop("medianFilter", None)
        self._dtype = "i2"        
    
    def _storeMedianFilter(self):
        if "medianFilter" in self.imageProcessing:
            fileName, fileFilter = QFileDialog.getSaveFileName(parent=None, caption="Save filter", 
                                                            directory=UserFileDirs.Root, 
                                                            filter="NumPy file (*.npy);;TIFF (*.tiff)")
            if fileName:
                selectedExt = re.search('\((.+?)\)', fileFilter).group(1).replace('*','')
                if not os.path.splitext(fileName)[1]:
                    fileName = fileName + selectedExt
                if "tiff" in selectedExt:
                    imwrite(fileName, self.imageProcessing["medianFilter"]["content"], dtype=self.imageProcessing["medianFilter"]["content"].dtype)
                else:
                    np.save(fileName, self.imageProcessing["medianFilter"]["content"])
