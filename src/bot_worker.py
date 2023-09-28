from ultralytics import YOLO
import cv2
import numpy as np
from windows import WindowManagerAbstract
from PyQt6.QtCore import QThread, QWaitCondition, QMutex, pyqtSignal
from typing import *

@final
class BotWorker(QThread):

    displayImageSignal = pyqtSignal(np.ndarray, str)

    def __init__(self, window_manager: WindowManagerAbstract,
                 model_detection_path: str):
        super().__init__()
        self.__wait_condition = QWaitCondition()
        self.__mutex = QMutex()
        self.__paused = True

        self.wm = window_manager
        self.debug_window = False

        self.__model = YOLO(model_detection_path)
        self.__threshold = 0.75

    def __draw_boxe(self, img: np.ndarray, x1: int, x2: int, y1: int, y2: int,
                    class_name: str) -> None:
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, class_name.upper(),
                    (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0), 3, cv2.LINE_AA)

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
            results = self.__model(img)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                class_name = results.names[int(class_id)]

                if score > self.__threshold:
                    if self.debug_window:
                        self.__draw_boxe(img, x1, x2, y1, y2, class_name)

            if self.debug_window:
                self.displayImageSignal.emit(img, "Pixus Debug")
