from ultralytics import YOLO
import cv2
import numpy as np
from windows import WindowManagerAbstract
from PyQt6.QtCore import QThread, QWaitCondition, QMutex, pyqtSignal
from typing import *


@final
class BotWorker(QThread):

    display_image_signal = pyqtSignal(np.ndarray, str)

    def __init__(self, window_manager: WindowManagerAbstract,
                 model_detection_path: str):
        super().__init__()
        self.__wait_condition = QWaitCondition()
        self.__mutex = QMutex()
        self.__paused = True
        self.__HEIGTH = 960
        self.__WIDTH = 1024

        self.wm = window_manager
        self.__debug_window = False

        self.__model = YOLO(model_detection_path)
        self.__threshold = 0.75

    def __draw_boxe(self, img: np.ndarray, x1: int, x2: int, y1: int, y2: int,
                    class_name: str) -> None:
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, class_name.upper(),
                    (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)

    def __resize_image(self, img):
        h, w, _ = img.shape
        longest_edge = max(h, w)
        top_padding = (longest_edge - h) // 2
        bottom_padding = longest_edge - h - top_padding
        left_padding = (longest_edge - w) // 2
        right_padding = longest_edge - w - left_padding
        padded_image = cv2.copyMakeBorder(img, top_padding, bottom_padding,
                                          left_padding, right_padding,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized_image = cv2.resize(padded_image, (self.__WIDTH, self.__HEIGTH))
        return resized_image

    def debug_window_on(self):
        self.__debug_window = True

    def debug_window_off(self):
        self.__debug_window = False

    def pause(self) -> None:
        self.__mutex.lock()
        self.__paused = True
        self.__mutex.unlock()

    def resume(self) -> None:
        self.__mutex.lock()
        self.__paused = False
        self.__mutex.unlock()
        self.__wait_condition.wakeAll()

    def run(self) -> None:
        while True:
            self.__mutex.lock()
            if self.__paused:
                self.__wait_condition.wait(self.__mutex)
            self.__mutex.unlock()

            img = self.wm.get_window_array()
            img = self.__resize_image(img)
            results = self.__model(img)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                center_x = x2 - x1 // 2
                center_y = y2 - y1 // 2
                class_name = results.names[int(class_id)]

                if score > self.__threshold:
                    if self.__debug_window:
                        self.__draw_boxe(img, x1, x2, y1, y2, class_name)

            if self.__debug_window:
                self.display_image_signal.emit(img, "Pixus Debug")
