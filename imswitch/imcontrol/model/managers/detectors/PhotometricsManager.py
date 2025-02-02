import numpy as np
import Pyro5.api
import re, os
from tifffile.tifffile import imwrite
from numba import vectorize, float32
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model.dirtools import UserFileDirs
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

class PhotometricsManager(DetectorManager):
    """ DetectorManager that deals with frame extraction for a Photometrics camera.

    Manager properties:

    - ``cameraListIndex`` -- the camera's index in the Photometrics camera list
      (list indexing starts at 0)
    - ``medianFilter`` -- dictionary of parameters required for median filter acquisition
    - ``server`` -- URI of local ImSwitchServer in the format of host-port (i.e. \"127.0.0.1:54333\") 
    """

    def __init__(self, detectorInfo, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)

        self._camera = self._getCameraObj(detectorInfo.managerProperties['cameraListIndex'])
        self._binning = 1

        fullShape = self._camera.sensor_size

        model = self._camera.name
            
        self.__acquisition = False
        
        self.__chunkFramesSize = 100
        
        # default exposure time resolution is in milliseconds
        # with this we set the default resolution to microseconds
        self._camera.exp_res = 1
        
        startExpTime = 1e-3
        
        # Prepare parameters
        parameters = {
            'Set exposure time': DetectorNumberParameter(group='Timings', value=startExpTime,
                                                         valueUnits='s', editable=True),
            'Real exposure time': DetectorNumberParameter(group='Timings', value=startExpTime,
                                                          valueUnits='s', editable=False),
            'Readout time': DetectorNumberParameter(group='Timings', value=0,
                                                    valueUnits='s', editable=False),
            'Frame rate': DetectorNumberParameter(group='Timings', value=0,
                                                  valueUnits='FPS', editable=False),
            'Trigger source': DetectorListParameter(group='Acquisition mode',
                                                    value=list(self._camera.exp_modes.keys())[0],
                                                    options=list(self._camera.exp_modes.keys()),
                                                    editable=True),
            'Expose out mode': DetectorListParameter(group='Expose out',
                                                    value=list(self._camera.exp_out_modes.keys())[0],
                                                    options=list(self._camera.exp_out_modes.keys()),
                                                    editable=True),
            'Readout port': DetectorListParameter(group='Ports',
                                                  value='Dynamic range',
                                                  options=['Dynamic range',
                                                           'Speed',
                                                           'Sensitivity'], editable=True),
            'Chunk of frames': DetectorNumberParameter(group='Recording', value=self.__chunkFramesSize, 
                                                       valueUnits="frames", editable=True),
            'Camera pixel size': DetectorNumberParameter(group='Miscellaneous', value=0.1,
                                                         valueUnits='µm', editable=True)
        }
        
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
        
        actions = {}
        if (self.__uri is not None
            and self.__mfPositioners is not None
            and self.__mfStep is not None
            and self.__mfMaxFrames is not None):
            
            parameters["Step size"] = DetectorNumberParameter(group="Median Filter", value=self.__mfStep, valueUnits="µm", editable=True)
            parameters["Number of frames"] = DetectorNumberParameter(group="Median Filter", value=self.__mfMaxFrames, valueUnits="", editable=True)

            actions["Generate median filter"] = DetectorAction(group="Median Filter", func=self._generateMedianFilter)
            parameters["Operation"] = DetectorListParameter(group="Median Filter", value="Division", options=["Division", "Subtraction"], editable=True)
            actions["Clear median filter"] = DetectorAction(group="Median Filter", func=self._clearMedianFilter)
            actions["Store median filter"] = DetectorAction(group="Median Filter", func=self._storeMedianFilter)
            self.__medianFilterOp = numba_matrix_division

        super().__init__(detectorInfo, name, fullShape=fullShape, supportedBinnings=[1, 2, 4],
                         model=model, parameters=parameters, croppable=True, actions=actions)
        self._updatePropertiesFromCamera()
        super().setParameter('Set exposure time', self.parameters['Real exposure time'].value)

    @property
    def pixelSizeUm(self):
        umxpx = self.parameters['Camera pixel size'].value
        return [1, umxpx, umxpx]

    def getLatestFrame(self):
        try:
            status = self._camera.check_frame_status()
            if status == "READOUT_NOT_ACTIVE":
                return self.image
            else:
                img = np.array(self._camera.poll_frame()[0]['pixel_data'])
                if "medianFilter" in self.imageProcessing:
                    img = self.__medianFilterOp(self.image.astype(np.float32), self.imageProcessing["medianFilter"]["content"].astype(np.float32))
                return img
        except RuntimeError:
            return self.image

    def getChunk(self):
        chunkFrames = [] 
        status = self._camera.check_frame_status()
        try:
            if not status == "READOUT_NOT_ACTIVE":
                for _ in range(self.__chunkFramesSize):
                    im = np.array(self._camera.poll_frame()[0]['pixel_data'])
                    if "medianFilter" in self.imageProcessing:
                        im = self.__medianFilterOp(im.astype(np.float32), self.imageProcessing["medianFilter"]["content"].astype(np.float32))
                    chunkFrames.append(im)
        except RuntimeError:
            pass
        return chunkFrames

    def flushBuffers(self):
        pass

    def crop(self, hpos, vpos, hsize, vsize):
        """Method to crop the frame read out by the camera. """
        def updateROI():
            self._camera.set_roi(hpos, vpos, hsize, vsize)
        # This should be the only place where self.frameStart is changedim
        self._frameStart = (hpos, vpos)
        # Only place self.shapes is changed
        self._shape = (hsize, vsize)
        self._performSafeCameraAction(updateROI)
        newReadoutTime = float(self._camera.readout_time * 1e-6)
        super().setParameter('Readout time', newReadoutTime)
        super().setParameter('Frame rate', round(1.0 / newReadoutTime, 2))

    def setBinning(self, binning):
        super().setBinning(binning)
        def updateBinning():
            self._camera.binning = binning
        self._performSafeCameraAction(updateBinning)

    def setParameter(self, name, value):
        super().setParameter(name, value)
        
        if name == 'Chunk of frames':
            self.__chunkFramesSize = int(value)
        elif name == 'Set exposure time':
            self._setExposure(value)
            self._updatePropertiesFromCamera()
        elif name == 'Trigger source':
            self._setTriggerSource(value)
        elif name == 'Readout port':
            self._setReadoutPort(value)
        elif name == 'Expose out mode':
            self._setExposeOutMode(value)
        return self.parameters

    def startAcquisition(self):
        self.__acquisition = True
        self._camera.start_live()

    def stopAcquisition(self):
        self.__acquisition = False
        self._camera.abort()
        self._camera.finish()

    def _setExposure(self, time):
        
        self._camera.exp_time = int(time * 1e6)
        
        def updateRealExposureTime():
            # needed only to update real exposure time
            # for proper visualization
            pass
        
        self._performSafeCameraAction(updateRealExposureTime)
        newReadoutTime = float(self._camera.readout_time * 1e-6)
        super().setParameter('Real exposure time', float(self._camera.last_exp_time * 1e-6))
        super().setParameter('Readout time', newReadoutTime)
        super().setParameter('Frame rate', round(1.0 / newReadoutTime, 2))
    
    def _setExposeOutMode(self, mode):
        self.__logger.debug(f"Change expose out mode {mode}")
        def updateExposeOutMode():
            self._camera.exp_out_mode = self._camera.exp_out_modes[mode]
        self._performSafeCameraAction(updateExposeOutMode)

    def _setTriggerSource(self, source):
        self.__logger.debug(f"Change trigger source to {source}")
        def updateTrigger():
            self._camera.exp_mode = source    
        self._performSafeCameraAction(updateTrigger)

    def _setReadoutPort(self, port):
        self.__logger.debug(f"Change readout port to {port}")

        def updatePort():
            if port == 'Sensitivity':
                port_value = 0
            elif port == 'Speed':
                port_value = 1
            elif port == 'Dynamic range':
                port_value = 2
            else:
                raise ValueError(f'Invalid readout port "{port}"')
            self._camera.readout_port = port_value
        
        self._performSafeCameraAction(updatePort)
        newReadoutTime = float(self._camera.readout_time * 1e-6)
        super().setParameter('Readout time', newReadoutTime)
        super().setParameter('Frame rate', round(1.0 / newReadoutTime, 2))

    def _updatePropertiesFromCamera(self):
        newReadoutTime = float(self._camera.readout_time * 1e-6)
        super().setParameter('Readout time', newReadoutTime)
        super().setParameter('Frame rate', round(1.0 / newReadoutTime, 2))
        readoutPort = self._camera.readout_port
        if readoutPort == 0:
            self.setParameter('Readout port', 'Sensitivity')
        elif readoutPort == 1:
            self.setParameter('Readout port', 'Speed')
        elif readoutPort == 2:
            self.setParameter('Readout port', 'Dynamic range')
    
    def _performSafeCameraAction(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        if self.__acquisition:
            self.stopAcquisition()
            function()
            self.startAcquisition()
        else:
            function()

    def finalize(self):
        self._camera.close()

    def _getCameraObj(self, cameraId):
        try:
            from pyvcam import pvc
            from pyvcam.camera import Camera

            pvc.init_pvcam()
            self.__logger.debug(f'Trying to initialize Photometrics camera {cameraId}')
            camera = next(Camera.detect_camera())
            camera.open()
        except Exception:
            self.__logger.warning(f'Failed to initialize Photometrics camera {cameraId},'
                                  f' loading mocker')
            from imswitch.imcontrol.model.interfaces import MockHamamatsu
            camera = MockHamamatsu()

        self.__logger.info(f'Initialized camera, model: {camera.name}')
        return camera

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



# Copyright (C) 2020-2021 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
