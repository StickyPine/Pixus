from Xlib import X, display as Xdisplay
from abc import ABC, abstractmethod
import numpy as np
import cv2
import pygetwindow as gw
import win32gui, win32con, win32ui
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
        return abs_geom.x+x, abs_geom.y+y

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

@final
class WindowManagerWindows(WindowManagerAbstract):

    def __init__(self, window_title: str):
        self.__window_title = window_title
        self.__window = None
        self.hwnd = None
        self.grab_window()
        # to crop the window border and titlebar off the image
        self.cropped_x = 8
        self.cropped_y = 30

    def grab_window(self):
        window_names = gw.getAllTitles()
        matching_names = [name for name in window_names if self.__window_title in name]
        if (len(matching_names) == 0):
            raise ValueError(f"Window with title '{self.__window_title}' not found.")
        self.hwnd = win32gui.FindWindow(None, matching_names[0])
        self.__window = gw.getWindowsWithTitle(matching_names[0])[0]        

    def translate_position(self, x: int, y: int) -> Tuple[int, int]:
        left, top, _, _ = self.__window.left, self.__window.top, self.__window.width, self.__window.height
        return left + x + self.cropped_x, top + y + self.cropped_y

    def resize_window(self, width: int, height: int) -> None:
        self.__window.resizeTo(width, height)

    def get_window_array(self) -> np.ndarray:
        
        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        # account for the window border and titlebar and cut them off
        self.w = self.w - (self.cropped_x * 2)
        self.h = self.h - self.cropped_y - self.cropped_x
        
        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a NumPy array
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        img = img[...,:3]
        img = np.ascontiguousarray(img)
        return img
    