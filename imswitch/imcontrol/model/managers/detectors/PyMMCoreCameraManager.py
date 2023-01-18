from ..PyMMCoreManager import PyMMCoreManager # only for type hinting
from imswitch.imcommon.model import initLogger

from .DetectorManager import (
    DetectorManager, DetectorNumberParameter
)

class PyMMCoreCameraManager(DetectorManager):
    def __init__(self, detectorInfo, name, **lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self.__coreManager: PyMMCoreManager = lowLevelManagers["pymmcManager"]

        module = detectorInfo.managerProperties["module"]
        device = detectorInfo.managerProperties["device"]
        
        devInfo = (name, module, device)

        self.__coreManager.loadDevice(devInfo, True)

        # todo: dictionary should be filled automatically from the core
        # by reading the available camera parameters and returning them as a dictionary

        properties = self.__coreManager.loadProperties(name)

        parameters = {
            'Exposure': DetectorNumberParameter(group='Timings', value=10,
                                                valueUnits='ms', editable=True),
            'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=10,
                                                         valueUnits='µm', editable=True), 
        }
        _, _, hsize, vsize = self.__coreManager.getROI(name)

        super().__init__(detectorInfo, name, fullShape=(hsize, vsize), supportedBinnings=[1], 
                        model=device, parameters=parameters, croppable=True)
        self.setParameter("Exposure", self.parameters["Exposure"].value)
        self.__frame = None
    
    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]
    
    def setBinning(self, binning):
        # todo: Ximea binning works in a different way
        # investigate how to properly update this value
        # if possible
        super().setBinning(binning)
    
    def getLatestFrame(self, is_save=False):
        if self.__coreManager.coreObject.getRemainingImageCount() > 0:
            self.__frame = self.__coreManager.coreObject.popNextImage()
        return self.__frame
    
    def getChunk(self):
        return [self.getLatestFrame()]
    
    def flushBuffers(self):
        self.__coreManager.coreObject.clearCircularBuffer()
    
    def startAcquisition(self):
        self.__coreManager.coreObject.startContinuousSequenceAcquisition(self.parameters["Exposure"].value)

    def stopAcquisition(self):
        self.__coreManager.coreObject.stopSequenceAcquisition(self.name)

    
    def setParameter(self, name, value):
        # this may not work properly, keep an eye on it
        if name == "Exposure":
            self.__coreManager.coreObject.setExposure(value)
        else:
            self.__coreManager.setProperty(self.name, name, value) 
        # there will be still images left in the circular buffer
        # captured using the previous property value, so we flush the buffer
        self.flushBuffers()
        super().setParameter(name, value)

    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int):
        self.__coreManager.setROI(self.name, hpos, vpos, hsize, vsize)
        
        # This should be the only place where self.frameStart is changed
        self._frameStart = (hpos, vpos)
        # Only place self.shapes is changed
        self._shape = (hsize, vsize)