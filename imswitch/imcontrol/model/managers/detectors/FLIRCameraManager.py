
import numpy as np
import PySpin
from typing import Dict, Any
from dataclasses import dataclass
from imswitch.imcommon.model import initLogger
from contextlib import contextmanager
from .DetectorManager import (
    DetectorManager, DetectorNumberParameter, DetectorListParameter, DetectorParameter
)

class FLIRCameraManager(DetectorManager):

    # supported parameters
    acquisitionMode = {
        "Continous" : PySpin.AcquisitionMode_Continuous, 
        "Single frame": PySpin.AcquisitionMode_SingleFrame,
        "Multi frame": PySpin.AcquisitionMode_MultiFrame
    }
    
    triggerSelector = {
        "Acquisition start" : PySpin.TriggerSelector_AcquisitionStart,
        "Frame start": PySpin.TriggerSelector_FrameStart,
        "Frame burst start" : PySpin.TriggerSelector_FrameBurstStart
    }

    pixelFormat = {
        "Mono8" : PySpin.PixelFormat_Mono8,
        "Mono16" : PySpin.PixelFormat_Mono16,

        # these are also supported formats but they need to be
        # converted with the QuickSpin APIs somehow
        # "Mono10 packed" : PySpin.PixelFormat_Mono10Packed,
        # "Mono12 packed" : PySpin.PixelFormat_Mono12Packed
    }

    ADCBitDepth = {
        "10 bit": PySpin.AdcBitDepth_Bit10,
        "12 bit": PySpin.AdcBitDepth_Bit12
    }

    def __init__(self, detectorInfo, name, **_lowLevelManagers) -> None:
        
        self.__logger = initLogger(self, instanceName=name)
        self.__system = PySpin.System.GetInstance()
        
        version = self.__system.GetLibraryVersion()
        self.__logger.info(f"PySpin library version: {version.major}.{version.minor}.{version.type}.{version.build}")
        camIndex = detectorInfo.digitalLine

        # we need to keep the camera list as a local object
        # then we release it after finishing initialization of the camera
        # otherwise the system object will consider it a dangling pointer
        # and won't be able to exit gracefully
        camera_list = self.__system.GetCameras()


        if camera_list.GetSize() > 0:
            try:
                self.__camera = camera_list.GetByIndex(camIndex)
            except:
                error_msg = f"No FLIR camera found at index {camIndex}"
                camera_list.Clear()
                self.__system.ReleaseInstance()
                self.__logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = "No FLIR camera found!"
            camera_list.Clear()
            self.__system.ReleaseInstance()
            self.__logger.error(error_msg)
            raise ValueError(error_msg)        
        # after we make sure the camera is initialized,
        # we clear the camera_list pointer
        self.__camera.Init()
        camera_list.Clear()

        # TODO: make this a parameter
        self.__readTimeout = 100

        parameters = {
            "Exposure" : DetectorNumberParameter(group="Timings", value=100e-6, editable=True, valueUnits="s"),
            "Acquisition mode" : DetectorListParameter(group="Trigger settings", 
                                                        value="Continous", 
                                                        editable=True, 
                                                        options=list(self.acquisitionMode.keys())),
            "Trigger selector" : DetectorListParameter(group="Trigger settings",
                                                        value="Acquisition start",
                                                        editable=True,
                                                        options=list(self.triggerSelector.keys())),
            "Pixel format" : DetectorListParameter(group="Device settings", 
                                                    value="Mono16", 
                                                    editable=True,
                                                    options=list(self.pixelFormat.keys())),
            "ADC bit depth" : DetectorListParameter(group="Device settings",
                                                    value="12 bit",
                                                    editable=True,
                                                    options=list(self.ADCBitDepth.keys())),
            "Pixel size" : DetectorNumberParameter(group="Miscellaneous", value=3.45, editable=True, valueUnits="µm"),
        }

        fullShape = (self.__camera.Width.GetMax(), self.__camera.Height.GetMax())

        # disable auto exposure
        self.__camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

        # set default configuration at startup
        self.__camera.OffsetX.SetValue(0)
        self.__camera.OffsetY.SetValue(0)
        self.__camera.Width.SetValue(self.__camera.Width.GetMax())
        self.__camera.Height.SetValue(self.__camera.Height.GetMax())
        self.__camera.ExposureTime.SetValue(100) # microseconds
        self.__camera.AcquisitionMode.SetValue(self.acquisitionMode["Continous"])
        self.__camera.TriggerSource.SetValue(self.triggerSelector["Acquisition start"])
        self.__camera.PixelFormat.SetValue(self.pixelFormat["Mono16"])
        self.__camera.AdcBitDepth.SetValue(self.ADCBitDepth["12 bit"])

        model = self.__camera.TLDevice.DeviceModelName.GetValue()

        super().__init__(detectorInfo, name=name, 
                        fullShape=fullShape, 
                        model=model, 
                        parameters=parameters, 
                        supportedBinnings=[1],
                        croppable=True)

    @contextmanager
    def _camera_disabled(self):
        if self.__camera.IsStreaming():
            try:
                self.stopAcquisition()
                yield
            finally:
                self.startAcquisition()
        else:
            yield
    
    @contextmanager
    def _camera_enabled(self):
        if not self.__camera.IsStreaming():
            try:
                self.startAcquisition()
                yield
            finally:
                self.stopAcquisition()
        else:
            yield
    
    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Pixel size'].value
        return [1, umxpx, umxpx]
    
    def getLatestFrame(self, is_save=False) -> np.ndarray:
        img = self.__camera.GetNextImage(self.__readTimeout)
        data = img.GetNDArray()
        img.Release()
        return data
    
    def setBinning(self, binning):
        # TODO: investigate how binning works...
        super().setBinning(binning)
    
    def setParameter(self, name: str, value: Any) -> Dict[str, DetectorParameter]:
        with self._camera_disabled():
            if name == "Exposure":
                # make sure to clip exposure time value
                value = min(self.__camera.ExposureTime.GetMax(), max(self.__camera.ExposureTime.GetMin(), int(value*1e6)))
                self.__camera.ExposureTime.SetValue(value)
            elif name == "Acquisition mode":
                self.__camera.AcquisitionMode.SetValue(self.acquisitionMode[value])
            elif name == "Trigger selector":
                self.__camera.TriggerSource.SetValue(self.triggerSelector[value])
            elif name == "Pixel format":
                self.__camera.PixelFormat.SetValue(self.pixelFormat[value])
            elif name == "ADC bit depth":
                self.__camera.AdcBitDepth.SetValue(self.ADCBitDepth[value])
            else:
                self.__logger.warning(f"No option available for \"{name}\", skipping")
        
        return super().setParameter(name, value)
    
    def getChunk(self) -> np.ndarray:
        # TODO: there should be a way to get more images at the same time
        # either from API or using a local ring buffer
        return np.stack([self.getLatestFrame()])
    
    def flushBuffers(self) -> None:
        # TODO: find appropriate implementation
        # if not, leave as it is
        pass
    
    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:

        if hpos + hsize > self.__camera.Width.GetMax():
            self.__logger.error(f"Horizontal positions invalid (offset: {hpos}, width: {hsize}); aborting")
            return
        
        if vpos + vsize > self.__camera.Height.GetMax():
            self.__logger.error(f"Vertical positions invalid (offset: {vpos}, height: {vsize}; aborting")
            return
        
        with self._camera_disabled():
            self.__camera.OffsetX.SetValue(hpos)
            self.__camera.OffsetY.SetValue(vpos)
            self.__camera.Width.SetValue(hsize)
            self.__camera.Height.SetValue(vsize)
        
        self._frameStart = (hpos, vpos)
        self._shape = (hsize, vsize)
    
    def startAcquisition(self) -> None:
        self.__camera.BeginAcquisition()
    
    def stopAcquisition(self) -> None:
        self.__camera.EndAcquisition()

    def finalize(self) -> None:
        if self.__camera.IsStreaming():
            self.__camera.EndAcquisition()
        self.__camera.DeInit()
        del self.__camera
        self.__system.ReleaseInstance()
