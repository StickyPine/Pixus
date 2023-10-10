from .window_handler_abstract import WindowHandlerAbstract
import pygetwindow as gw
import win32gui
import win32con
import win32ui
import numpy as np
from typing import *


@final
class WindowsHandler(WindowHandlerAbstract):

    def __init__(self, window_title: str):
        super().__init__(window_title)

        self.__window = None
        self.hwnd = None
        self.grab_window()
        # to crop the window border and titlebar off the image
        self.cropped_x = 8
        self.cropped_y = 30

    def grab_window(self):
        window_names = gw.getAllTitles()
        matching_names = [name for name in window_names if self._window_name in name]
        if (len(matching_names) == 0):
            raise ValueError(f"Window with title '{self._window_name}' not found.")
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

        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img
