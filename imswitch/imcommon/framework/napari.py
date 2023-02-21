from napari.qt.threading import thread_worker, WorkerBaseSignals, FunctionWorker
from .qt import Signal

class RecordingSignals(WorkerBaseSignals):
    frameNumberUpdate = Signal(int)
    timeUpdate = Signal(float)

class StreamingWorker(FunctionWorker):
    def __init__(self, func, *args, **kwargs):
        super().__init__(func, *args, **kwargs, SignalsClass=WorkerBaseSignals)