"""
Video capture and FPS tracking utilities.
"""
import cv2
import time
import numpy as np
from threading import Thread


class WebcamVideoStream:
    """Threaded webcam capture for better FPS.

    Based on: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """
    def __init__(self, src: int = 0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self) -> "WebcamVideoStream":
        Thread(target=self._update, daemon=True).start()
        return self

    def _update(self) -> None:
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self) -> np.ndarray:
        return self.frame

    def stop(self) -> None:
        self.stopped = True
        self.stream.release()


class FPSCounter:
    """Simple FPS counter with rolling average."""
    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.times: list[float] = []
        self.fps = 0.0

    def tick(self) -> float:
        self.times.append(time.time())
        if len(self.times) > self.avg_frames:
            self.times.pop(0)
        if len(self.times) >= 2:
            self.fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])
        return self.fps

    def get(self) -> float:
        return self.fps
