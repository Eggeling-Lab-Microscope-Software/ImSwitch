import enum
import os
import time
from math import ceil
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Type
from types import DynamicClassAttribute
from imswitch.imcontrol.model.configfiletools import _debugLogDir

import h5py
import zarr
import numpy as np
import tifffile as tiff

from imswitch.imcommon.framework import (
    Signal, 
    SignalInterface,
    FunctionWorker,
    create_worker
)
from imswitch.imcommon.model import initLogger
import logging

from imswitch.imcontrol.model.managers.DetectorsManager import DetectorsManager
from imswitch.imcontrol.model.managers.detectors.DetectorManager import DetectorManager

logger = logging.getLogger(__name__)

class StreamingWorker(FunctionWorker):
    pass

class RecMode(enum.Enum):
    SpecFrames = 1
    SpecTime = 2
    ScanOnce = 3
    ScanLapse = 4
    UntilStop = 5

class SaveMode(enum.Enum):
    Disk = 1
    RAM = 2
    DiskAndRAM = 3
    Numpy = 4

class SaveFormat(enum.Enum):
    HDF5 = 1
    TIFF = 2
    ZARR = 3
    
    @DynamicClassAttribute
    def name(self):
        name = super(SaveFormat, self).name
        if name == "TIFF":
            name = "OME-TIFF"
        return name

class AsTemporayFile(object):
    """ A temporary file that when exiting the context manager is renamed to its original name. """

    def __init__(self, filepath, tmp_extension='.tmp'):
        if os.path.exists(filepath):
            raise FileExistsError(f'File {filepath} already exists.')
        self.path = filepath
        self.tmp_path = filepath + tmp_extension

    def __enter__(self):
        return self.tmp_path

    def __exit__(self, *args, **kwargs):
        os.rename(self.tmp_path, self.path)
        logger.info("Renamed file from %s to %s", self.tmp_path, self.path)



class Storer(SignalInterface):
    """ Base class for storing data """
    frameNumberUpdate = Signal(str, int) # channel, frameNumber
    timeUpdate = Signal(str, float) # channel, timeCount

    def __init__(self, filepath: str, detectorsManager: DetectorsManager):
        super().__init__()
        self.filepath = filepath
        self.detectorsManager = detectorsManager
        self.frameCount : int = 0
        self.timeCount : float = 0.0
        self.__record = False
    
    @property
    def record(self) -> bool:
        return self.__record

    @record.setter
    def record(self, record: bool):
        self.__record = record

    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        """ Stores images and attributes according to the spec of the storer """
        raise NotImplementedError

    def stream(self, channel: str, recMode: RecMode, attrs: Dict[str, str], **kwargs) -> None:
        """ Stores data in a streaming fashion. """
        raise NotImplementedError
    
    def unpackChunk(self, detector: DetectorManager) -> Tuple[np.ndarray, np.ndarray]:
        """ Checks if the value returned by getChunk is a tuple (packing the recorded chunk and the associated time points).
        If the return type is only the recorded stack, a zero array is generated with the same length of 
        the recorded chunk.

        Args:
            detector (DetectorManager): detector to read chunk from

        Returns:
            tuple: a 2-element tuple with the recorded chunk and the single data points associated with each frame.
        """
        chunk = detector.getChunk()
        if type(chunk) == tuple:
            return chunk
        else: # type is np.ndarray
            chunk = (chunk, np.zeros(len(chunk)))
        return chunk



class ZarrStorer(Storer):
    """ A storer that stores the images in a zarr file store """
    
    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):

        with AsTemporayFile(f'{self.filepath}.zarr') as path:
            store = zarr.storage.DirectoryStore(path)
            root = zarr.group(store=store)

            for channel, image in images.items():
                shape = self.detectorsManager[channel].shape

                d = root.create_dataset(channel, data=image, shape=tuple(reversed(shape)),
                                        chunks=(512, 512), dtype='i2') #TODO: why not dynamic chunking?
                d.attrs["ImSwitchData"] = attrs[channel]
                logger.info(f"Saved image to zarr file {path}")


class HDF5Storer(Storer):
    """ A storer that stores the images in a series of hd5 files """

    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):

        for channel, image in images.items():
            
            with AsTemporayFile(f'{self.filepath}_{channel}.h5') as path:
                file = h5py.File(path, 'w')

                shape = self.detectorsManager[channel].shape

                dataset = file.create_dataset('data', tuple(reversed(shape)), dtype='i2')

                for key, value in attrs[channel].items():
                    try:
                        dataset.attrs[key] = value
                    except:
                        logger.debug(f'Could not put key:value pair {key}:{value} in hdf5 metadata.')

                dataset.attrs['detector_name'] = channel

                # For ImageJ compatibility
                dataset.attrs['element_size_um'] = \
                    self.detectorsManager[channel].pixelSizeUm

                dataset[:, ...] = np.moveaxis(image, 0, -1)
            
                file.close()

    def stream(self, channel: str, recMode: RecMode, attrs: Dict[str, str], **kwargs) -> None:
        detector : DetectorManager = self.detectorsManager[channel]
        pixelSize = detector.pixelSizeUm
        self.record = True
        frameTimePoints = []
        
        def create_dataset(file: h5py.File, shape: tuple) -> h5py.Dataset:
            dataset = file.create_dataset("data", shape=shape, maxshape=(None, *shape[1:]), dtype=detector.dtype)
            for key, value in attrs.items():
                try:
                    dataset.attrs[key] = value
                except:
                    logger.debug(f'Could not put key:value pair {key}:{value} in hdf5 metadata.')
            dataset.attrs["detector_name"] = channel
            dataset.attrs["element_size_um"] = pixelSize
            return dataset
                
        with h5py.File(self.filepath, mode="w") as file:
            if recMode in [RecMode.SpecFrames, RecMode.ScanLapse]:
                totalFrames = kwargs["totalFrames"]
                self.frameCount = 0
                dataset = create_dataset(file, (totalFrames, *reversed(detector.shape)))
                while self.frameCount < totalFrames and self.record:
                    frames, timePoints = self.unpackChunk(detector)
                    if self.frameCount + len(frames) > totalFrames:
                        # we only collect the remaining frames required,
                        # and discard the remaining
                        frames = frames[0: totalFrames - self.frameCount]
                        timePoints = timePoints[0: totalFrames - self.frameCount]
                    dataset[self.frameCount : len(frames)] = frames
                    frameTimePoints.extend(timePoints)
                    self.frameCount += len(frames)
                    self.frameNumberUpdate.emit(channel, self.frameCount)
                dataset.attrs["ΔT"] = frameTimePoints
            elif recMode == RecMode.SpecTime:
                timeUnit = detector.frameInterval # us
                totalTime = kwargs["totalTime"] # s
                totalFrames = int(ceil(totalTime * 1e6 / timeUnit))
                dataset = create_dataset(file, (totalFrames, *reversed(detector.shape)))
                currentRecTime = 0
                start = time.time()
                index = 0
                while index < totalFrames and self.record:
                    frames, timePoints = self.unpackChunk(detector)
                    nframes = len(frames)
                    dataset[index: nframes] = frames
                    frameTimePoints.extend(timePoints)
                    self.timeCount = np.around(currentRecTime, decimals=2)
                    self.timeUpdate.emit(channel, min(self.timeCount, totalTime))
                    index += nframes
                    currentRecTime = time.time() - start
                dataset.attrs["ΔT"] = frameTimePoints
                # we may have not used up the entirety of the HDF5 size,
                # so we resize the dataset to the value of "index"
                # in case this is lower than totalFrames
                if index < totalFrames:
                    dataset.resize(index, axis=0)
            elif recMode == RecMode.UntilStop:
                # with HDF5 it's hard to make an estimation of an infinite recording
                # and set the correct data size... the best thing we can do is to 
                # create the dataset big enough to store 1 second worth of data recording
                # and keep extending it whenever we're going out of boundaries
                # but a better solution should be found
                timeUnit = detector.frameInterval # us
                totalTime = 1e6 # us
                totalFrames = int(ceil(totalTime / timeUnit))
                dataset = create_dataset(file, (totalFrames, *detector.shape))
                index = 0
                while self.record:
                    frames, timePoints = self.unpackChunk(detector)
                    nframes = len(frames)
                    if nframes > index:
                        dataset.resize(index + totalFrames, axis=0)                    
                    dataset[index: nframes] = frames
                    frameTimePoints.extend(timePoints)
                    index += nframes
                dataset.attrs["ΔT"] = frameTimePoints
                    
                
class TiffStorer(Storer):
    """ A storer that stores the images in a series of TIFF files. """

    def snap(self, images: Dict[str, np.ndarray], attrs: Dict[str, str] = None):
        for channel, image in images.items():
            with AsTemporayFile(f'{self.filepath}_{channel}.tiff') as path:
                tiff.imwrite(path, image,) # TODO: Parse metadata to tiff meta data
    
    def stream(self, channel: str, recMode: RecMode, attrs: Dict[str, str], **kwargs) -> None:
        # TODO: Parse metadata to tiff meta data
        detector: DetectorManager = self.detectorsManager[channel]
        _, physicalYSize, physicalXSize = self.detectorsManager[channel].pixelSizeUm
        # resolution = (physicalYSize, 1./physicalXSize)
        frameTimePoints = []
        
        self.record = True
        
        with tiff.TiffWriter(self.filepath, ome=False, bigtiff=True) as file:
            if recMode == RecMode.SpecFrames:
                totalFrames = kwargs["totalFrames"]
                while self.frameCount < totalFrames and self.record:
                    frames, timePoints = self.unpackChunk(detector)
                    if self.frameCount + len(frames) >= totalFrames:
                        # we only collect the remaining frames required,
                        # and discard the remaining
                        frames = frames[0: totalFrames - self.frameCount]
                        timePoints = timePoints[0: totalFrames - self.frameCount]
                    file.write(frames.reshape((len(frames), 1, *frames.shape[1:])),  photometric="minisblack", contiguous=True, description="", metadata=None) #, resolution=resolution)
                    frameTimePoints.extend(timePoints)
                    self.frameCount += len(frames)
                    self.frameNumberUpdate.emit(channel, self.frameCount)
            elif recMode == RecMode.SpecTime:
                timeUnit = detector.frameInterval # us
                totalTime = kwargs["totalTime"] # s
                totalFrames = int(ceil(totalTime * 1e6 / timeUnit))
                currentRecTime = 0
                start = time.time()
                index = 0
                while index < totalFrames and self.record:
                    frames, timePoints = self.unpackChunk(detector)
                    file.write(frames.reshape((len(frames), 1, *frames.shape[1:])), photometric="minisblack", contiguous=True, description="", metadata=None) #, resolution=resolution)
                    frameTimePoints.extend(timePoints)
                    self.timeCount = np.around(currentRecTime, decimals=2)
                    self.timeUpdate.emit(channel, min(self.timeCount, totalTime))
                    currentRecTime = time.time() - start
                self.frameCount = totalFrames
            elif recMode == RecMode.UntilStop:
                while self.record:
                    frames, timePoints = self.unpackChunk(detector)
                    file.write(frames.reshape((len(frames), 1, *frames.shape[1:])), photometric="minisblack", contiguous=True, description="", metadata=None) #, resolution=resolution)
                    frameTimePoints.extend(timePoints)
                    self.frameCount += len(frames)
        
            # after recording, we need to ensure that all time points
            # are coherent with the frame interval specified by the detector
            # we check if we are within a 1 μs tolerance range
            if not np.all(np.isclose(frameTimePoints[1:], detector.frameInterval, atol=1)):
                logger.error(f"Frames were lost. Writing {detector.frameInterval} μs as time interval.")
                with open(os.path.join(_debugLogDir, "tiff_recording_timepoints.log"), 'w') as f:
                    for item in frameTimePoints:
                        f.write(f"{item}\n")
            else:
                logger.info("No frames lost!")
                with open(os.path.join(_debugLogDir, "tiff_recording_timepoints.log"), 'w') as f:
                    for item in frameTimePoints:
                        f.write(f"{item}\n")
            omexml = tiff.OmeXml()
            omexml.addimage(
                dtype=detector.dtype,
                shape=(self.frameCount, *detector.shape),
                storedshape=(self.frameCount, 1, 1, *detector.shape, 1),
                axes='TYX',
                TimeIncrement=detector.frameInterval,
                TimeIncrementUnit='μs',
                PhysicalSizeX = physicalXSize,
                PhysicalSizeXUnit = 'µm',
                PhysicalSizeY = physicalYSize,
                PhysicalSizeYUnit = 'µm'
            )
            description = omexml.tostring(declaration=True).encode()
            file.overwrite_description(description)

class BytesIOStorer(Storer):
    """Storer class for local RAM data. Uses BytesIO to stream images from detectors
    to local memory buffer.
    """    
    def stream(self, channel: str, recMode: RecMode, attrs: Dict[str, str], **kwargs) -> None:
        # TODO: parse metadata... does it make sense?
        memRecording: BytesIO = kwargs["fileDests"][channel]
        detector: DetectorManager = self.detectorsManager[channel]
        
        self.record = True
        if recMode == RecMode.SpecFrames:
            if recMode == RecMode.SpecFrames:
                totalFrames = kwargs["totalFrames"]
                while self.frameCount < totalFrames and self.record:
                    frames = detector.getChunk()
                    if self.frameCount + len(frames) >= totalFrames:
                        # we only collect the remaining frames required,
                        # and discard the remaining
                        frames = frames[0: totalFrames - self.frameCount]
                    memRecording.write(frames)
                    self.frameCount += len(frames)
                    self.frameNumberUpdate.emit(channel, self.frameCount)
                return
            elif recMode == RecMode.SpecTime:
                totalTime = kwargs["totalTime"]
                currentRecTime = 0
                start = time.time()
                while currentRecTime < totalTime and self.record:
                    frames = detector.getChunk()
                    memRecording.write(frames)
                    self.timeCount = np.around(currentRecTime, decimals=2)
                    self.timeUpdate.emit(channel, self.timeCount)
                    currentRecTime = time.time() - start
            elif recMode == RecMode.UntilStop:
                while self.record:
                    frames = detector.getChunk()
                    memRecording.write(frames)

DEFAULT_STORER_MAP: Dict[str, Type[Storer]] = {
    SaveFormat.ZARR: ZarrStorer,
    SaveFormat.HDF5: HDF5Storer,
    SaveFormat.TIFF: TiffStorer
}

class RecordingManager(SignalInterface):
    """ RecordingManager handles single frame captures as well as continuous
    recordings of detector data. """

    sigRecordingStarted = Signal()
    sigRecordingEnded = Signal()
    sigRecordingFrameNumUpdated = Signal(int)  # (frameNumber)
    sigRecordingTimeUpdated = Signal(int)  # (recTime)
    sigMemorySnapAvailable = Signal(
        str, np.ndarray, object, bool
    )  # (name, image, filePath, savedToDisk)
    sigMemoryRecordingAvailable = Signal(
        str, object, object, bool
    )  # (name, file, filePath, savedToDisk)

    def __init__(self, detectorsManager, storerMap: Optional[Dict[str, Type[Storer]]] = None):
        super().__init__()
        self.__logger = initLogger(self)
        self.__storerMap: Dict[str, Type[Storer]] = storerMap or DEFAULT_STORER_MAP
        self._memRecordings: Dict[str, Type[BytesIO]] = {}  # { filePath: bytesIO }
        self.__detectorsManager : DetectorsManager = detectorsManager
        self.__record = False
        self.__signalBuffer = dict()
        self.__threadCount = 0
        self.__totalThreads = 0
        self.__recordingHandle = None
        self.__storersList : List[Storer] = []
        self.__storerThreads : List[StreamingWorker] = [] 

    def __del__(self):
        self.endRecording(emitSignal=False, wait=True)
        if hasattr(super(), '__del__'):
            super().__del__()

    @property
    def record(self):
        """ Whether a recording is currently being recorded. """
        return self.__record

    @property
    def detectorsManager(self):
        return self.__detectorsManager
    
    def getFiles(self,
                savename: str,
                detectorNames: List[str],
                recMode: RecMode,
                saveMode: SaveMode,
                saveFormat: SaveFormat,
                singleLapseFile: bool = False) -> Tuple[dict, dict]:
        singleLapseFile = recMode == RecMode.ScanLapse and singleLapseFile

        fileDests = dict()
        filePaths = dict()
        extension = saveFormat.name.replace("-", ".").lower()

        for detectorName in detectorNames:
            baseFilePath = f'{savename}_{detectorName}.{extension}'

            filePaths[detectorName] = self.getSaveFilePath(
                baseFilePath,
                allowOverwriteDisk=singleLapseFile and saveMode != SaveMode.RAM,
                allowOverwriteMem=singleLapseFile and saveMode == SaveMode.RAM
            )

        for detectorName in detectorNames:
            if saveMode == SaveMode.RAM:
                memRecordings = self._memRecordings
                if (filePaths[detectorName] not in memRecordings or memRecordings[filePaths[detectorName]].closed):
                    memRecordings[filePaths[detectorName]] = BytesIO()
                fileDests[detectorName] = memRecordings[filePaths[detectorName]]
            else:
                fileDests[detectorName] = filePaths[detectorName]
        
        return fileDests, filePaths
    
    def _updateMemoryRecordingListings(self, storer: Storer) -> None:
        # method is called only when recording is finished (a.k.a. when the thread closes)
        # as it is triggered by the "finished" signal
        filePath = storer.filepath
        if type(storer) == TiffStorer:
            file = tiff.memmap(filePath)
        elif type(storer) == HDF5Storer:
            file = h5py.File(filePath, "a")
        else:
            raise NotImplementedError("RAM storage mode is currently limited to TIFF and HDF5, other formats not supported!")
        name = os.path.basename(storer.filepath)
        self.sigMemoryRecordingAvailable.emit(
            name, file, filePath, True
        )

    def _updateFramesCounter(self, channel: str, frameNumber: int) -> None:
        self.__signalBuffer[channel] = frameNumber
        self.sigRecordingFrameNumUpdated.emit(min(list(self.__signalBuffer.values())))
    
    def _updateSecondsCounter(self, channel: str, seconds: float) -> None:
        self.__signalBuffer[channel] = seconds
        self.sigRecordingTimeUpdated.emit(min(list(self.__signalBuffer.values())))
    
    def _closeThread(self, storer: Storer, thread: StreamingWorker) -> None:
        storer.record = False
        thread.quit()
        self.__threadCount += 1
        if self.__threadCount == self.__totalThreads:
            self.sigRecordingEnded.emit()

    def startRecording(self, 
                       detectorNames: List[str], 
                       recMode: RecMode, 
                       savename: str, 
                       saveMode: SaveMode, 
                       attrs: Dict[str, str],
                       saveFormat: SaveFormat = SaveFormat.HDF5, 
                       singleMultiDetectorFile: bool = False, 
                       singleLapseFile: bool = False,
                       recFrames: int = None, 
                       recTime: float = None):
        """ Starts a recording with the specified detectors, recording mode,
        file name prefix and attributes to save to the recording per detector.
        In SpecFrames mode, recFrames (the number of frames) must be specified,
        and in SpecTime mode, recTime (the recording time in seconds) must be
        specified. """
        
        self.__totalThreads = len(detectorNames)
        self.__threadCount = 0

        fileDests, filePaths = self.getFiles(savename, detectorNames, recMode, saveMode, saveFormat)
        recOptions = dict(totalFrames=recFrames, totalTime=recTime)
        
        if saveMode in [SaveMode.Disk, SaveMode.DiskAndRAM]:            
            self.__storersList = [self.__storerMap[saveFormat](path, self.detectorsManager) for path in list(filePaths.values())]
            self.__storerThreads = [create_worker(storer.stream,
                                        channel,
                                        recMode, 
                                        attrs[channel], 
                                        **recOptions,
                                        _worker_class=StreamingWorker,
                                        _start_thread=False) for (channel, storer) in zip(detectorNames, self.__storersList)]            
            if saveMode == SaveMode.DiskAndRAM:
                for storer, thread in zip(self.__storersList, self.__storerThreads):
                    thread.returned.connect(
                        lambda: self._updateMemoryRecordingListings(storer)
                    )                    
        elif saveMode == SaveMode.RAM:
            recOptions["fileDests"] = fileDests
            
            self.__storersList = [BytesIOStorer(fileDest, self.detectorsManager) for fileDest in list(fileDests.values())]
            self.__storerThreads = [create_worker(storer.stream,
                                           channel,
                                           recMode,
                                           attrs[channel],
                                           **recOptions,
                                           _worker_class=StreamingWorker,
                                           _start_thread=False) for (channel, storer) in zip(detectorNames, self.__storersList)]
            for storer, thread in zip(self.__storersList, self.__storerThreads):
                thread.returned.connect(
                    lambda: self._updateMemoryRecordingListings(storer)
                )
            
        if recMode in [RecMode.SpecFrames, RecMode.ScanLapse]:            
            for storer in self.__storersList:
                storer.frameNumberUpdate.connect(self._updateFramesCounter)
        elif recMode == RecMode.SpecTime:
            for storer in self.__storersList:
                storer.timeUpdate.connect(self._updateSecondsCounter)
        
        for storer, thread in zip(self.__storersList, self.__storerThreads):
            thread.returned.connect(
                lambda: self._closeThread(storer, thread)
            )
            # connecting to handle possible exceptions at run time
            thread.finished.connect(
                lambda: self._closeThread(storer, thread)
            )
        
        self.__logger.info('Starting recording')
        self.__record = True
        
        self.__recordingHandle = self.__detectorsManager.startAcquisition()
        self.sigRecordingStarted.emit()
        
        for thread in self.__storerThreads:
            thread.start()

    def endRecording(self, emitSignal=True, wait=True):
        """ Ends the current recording. Unless emitSignal is false, the
        sigRecordingEnded signal will be emitted. Unless wait is False, this
        method will wait until the recording is complete before returning. """

        if self.__recordingHandle != None:
            self.__detectorsManager.stopAcquisition(self.__recordingHandle)
            self.__recordingHandle = None
            if self.__record:
                self.__logger.info('Stopping recording')
                self.__record = False
            self.__detectorsManager.execOnAll(lambda c: c.flushBuffers(), condition=lambda c: c.forAcquisition)


    def snap(self, detectorNames, savename, saveMode, saveFormat, attrs):
        """ Saves an image with the specified detectors to a file
        with the specified name prefix, save mode, file format and attributes
        to save to the capture per detector. """
        acqHandle = self.__detectorsManager.startAcquisition()

        try:
            images = {}

            # Acquire data
            for detectorName in detectorNames:
                images[detectorName] = self.__detectorsManager[detectorName].getLatestFrame(is_save=True)

            if saveFormat:
                storer = self.__storerMap[saveFormat]

                if saveMode == SaveMode.Disk or saveMode == SaveMode.DiskAndRAM:
                    # Save images to disk
                    store = storer(savename, self.__detectorsManager)
                    store.snap(images, attrs)


                if saveMode == SaveMode.RAM or saveMode == SaveMode.DiskAndRAM:
                    for channel, image in images.items():
                        name = os.path.basename(f'{savename}_{channel}')
                        self.sigMemorySnapAvailable.emit(name, image, savename, saveMode == SaveMode.DiskAndRAM)

        finally:
            self.__detectorsManager.stopAcquisition(acqHandle)
            if saveMode == SaveMode.Numpy:
                return images

    def snapImagePrev(self, detectorName, savename, saveFormat, image, attrs):
        """ Saves a previously taken image to a file with the specified name prefix,
        file format and attributes to save to the capture per detector. """
        fileExtension = str(saveFormat.name).lower()
        filePath = self.getSaveFilePath(f'{savename}_{detectorName}.{fileExtension}')

        # Write file
        if saveFormat == SaveFormat.HDF5:
            file = h5py.File(filePath, 'w')

            shape = image.shape
            dataset = file.create_dataset('data', tuple(reversed(shape)), dtype='i2')

            for key, value in attrs[detectorName].items():
                try:
                    dataset.attrs[key] = value
                except:
                    self.__logger.debug(f'Could not put key:value pair {key}:{value} in hdf5 metadata.')

            dataset.attrs['detector_name'] = detectorName

            # For ImageJ compatibility
            dataset.attrs['element_size_um'] = \
                self.__detectorsManager[detectorName].pixelSizeUm

            dataset[:, ...] = np.moveaxis(image, 0, -1)
            file.close()
        elif saveFormat == SaveFormat.TIFF:
            tiff.imwrite(filePath, image)
        if saveFormat == SaveFormat.ZARR:
            path = self.getSaveFilePath(f'{savename}.{fileExtension}')
            store = zarr.storage.DirectoryStore(path)
            root = zarr.group(store=store)
            shape = self.__detectorsManager[detectorName].shape
            d = root.create_dataset(detectorName, data=image, shape=tuple(reversed(shape)), chunks=(512, 512),
                                    dtype='i2')
            d.attrs["ImSwitchData"] = attrs[detectorName]
            store.close()
        else:
            raise ValueError(f'Unsupported save format "{saveFormat}"')

    def getSaveFilePath(self, path, allowOverwriteDisk=False, allowOverwriteMem=False):
        newPath = path
        numExisting = 0

        def existsFunc(pathToCheck):
            if not allowOverwriteDisk and os.path.exists(pathToCheck):
                return True
            if not allowOverwriteMem and pathToCheck in self._memRecordings:
                return True
            return False

        while existsFunc(newPath):
            numExisting += 1
            pathWithoutExt, pathExt = os.path.splitext(path)
            newPath = f'{pathWithoutExt}_{numExisting}{pathExt}'
        return newPath

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
