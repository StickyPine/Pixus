from Xlib import X, display as Xdisplay
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import *


class WindowManagerAbstract(ABC):
    """Provide an OS/WM independent window abstraction"""

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


@final
class WindowManagerX11(WindowManagerAbstract):

    def __init__(self, window_class_name: str):
        self.__display = Xdisplay.Display()
        self.__window_class_name = window_class_name
        self.__window = None
        self.grab_window()

    def __get_absolute_geometry(self, geom):
        return (
            self.__display
            .screen()
            .root.translate_coords(self.__window,
                                   -geom.border_width,
                                   -geom.border_width)
        )

    def grab_window(self):
        root = self.__display.screen().root
        net_client_list = self.__display.intern_atom('_NET_CLIENT_LIST')
        window_ids = root.get_full_property(net_client_list,
                                            X.AnyPropertyType).value

        for window_id in window_ids:
            window = self.__display.create_resource_object('window', window_id)
            class_hint = window.get_wm_class()
            if class_hint and class_hint[1] == self.__window_class_name:
                self.__window = window

    def translate_position(self, x: int, y: int) -> Tuple[int, int]:
        geom = self.__window.get_geometry()
        abs_geom = self.__get_absolute_geometry(geom)
        return abs_geom.x+x, abs_geom+y

    def resize_window(self, width: int, height: int) -> None:
        self.__window.configure(width=width, height=height)
        self.__display.sync()

    def get_window_array(self) -> np.ndarray:
        geom = self.__window.get_geometry()
        raw = self.__window.get_image(0, 0, geom.width,
                                      geom.height, X.ZPixmap, 0xffffffff)
        image = np.frombuffer(raw.data, dtype=np.uint8)
        image.shape = (geom.height, geom.width, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

