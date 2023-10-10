from .window_handler_abstract import WindowHandlerAbstract
import win32gui
import win32con
import win32ui
import win32api
import numpy as np
from typing import *


@final
class WindowsHandler(WindowHandlerAbstract):

    def __init__(self, window_title: str):
        super().__init__(window_title)

        self.hwnd = None
        self.grab_window()

        # window borders
        self.w_border = win32api.GetSystemMetrics(win32con.SM_CXSIZEFRAME)
        self.h_border = win32api.GetSystemMetrics(win32con.SM_CYSIZEFRAME)
        self.titlebar = win32api.GetSystemMetrics(win32con.SM_CYCAPTION)
        
        self.h = None
        self.w = None

    def grab_window(self):

        def window_enum_callback(hwnd, _):
            window_title = win32gui.GetWindowText(hwnd)
            if self._window_name in window_title:
                matching_window.append(hwnd)

        matching_window = []
        win32gui.EnumWindows(window_enum_callback, None)    # loop through all windows
        if len(matching_window) == 0:
            raise ValueError(f"Window with title '{self._window_name}' not found")
        self.hwnd = matching_window[0]

    def translate_position(self, x: int, y: int) -> Tuple[int, int]:
        return win32gui.ClientToScreen(self.hwnd, (x, y))


    def resize_window(self, width: int, height: int) -> None:
        _, _, _, _, pos = win32gui.GetWindowPlacement(self.hwnd)    # get actual position
        win32gui.MoveWindow(self.hwnd, pos[0], pos[1], width + 2*self.w_border, height + 2*self.h_border + self.titlebar, True) # resize window


    def get_window_array(self) -> np.ndarray:

        # get window dims
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        h = bottom - top
        w = right - left
        
        if (h != 0 and w != 0): # update window dims if they are not 0
            self.h = bottom - top
            self.w = right - left
        # keep previous dims if they are 0 (window minimized)

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.w_border, self.h_border + self.titlebar), win32con.SRCCOPY)

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