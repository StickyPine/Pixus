from abc import ABC, abstractmethod
import numpy as np
from typing import *


class WindowHandlerAbstract(ABC):
    """Provide an OS/WM independent window abstraction"""

    def __init__(self, window_name: str):
        self._window_name = window_name

    @abstractmethod
    def translate_position(self, x: int, y: int) -> Tuple[int, int]:
        """Translate relative x,y coordinate to absolute display coordinate"""
        pass

    @abstractmethod
    def grab_window(self) -> None:
        """Grab the already present target window"""
        pass

    @abstractmethod
    def get_window_array(self) -> np.ndarray:
        """Return a numpy array image type that can be used for detection"""

    @abstractmethod
    def resize_window(self, width: int, height: int) -> None:
        """Resize the window to the given dimensions"""
        pass
