from .DetectorManager import DetectorManager
from ..PyMMCoreManager import PyMMCoreManager # only for type hinting
from imswitch.imcommon.model import initLogger

class PyMMCoreCameraManager(DetectorManager):
    def __init__(self, detectorInfo, name, **lowLevelManagers):
        self.__logger = initLogger()
        self.__coreManager: PyMMCoreManager = lowLevelManagers["pymmcManager"]

        module = detectorInfo.managerProperties["module"]
        device = detectorInfo.managerProperties["device"]
        
        devInfo = (name, module, device)
        self.__coreManager.loadDevice(devInfo)
    
    def getLatestFrame(self, is_save=False):
        return super().getLatestFrame(is_save)
    
    def getChunk(self):
        return super().getChunk()
    
    def flushBuffers(self):
        return super().flushBuffers()
    
    def startAcquisition(self):
        return super().startAcquisition()
    
    def stopAcquisition(self):
        return super().stopAcquisition()
    
    def setParameter(self, name, value):
        return super().setParameter(name, value)

    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int):
        return super().crop(hpos, vpos, hsize, vsize)
    
    def finalize(self):
        return super().finalize()